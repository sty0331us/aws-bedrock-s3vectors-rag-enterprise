import * as path from "node:path";
import * as cdk from "aws-cdk-lib";
import * as apigwv2 from "aws-cdk-lib/aws-apigatewayv2";
import * as apigwv2i from "aws-cdk-lib/aws-apigatewayv2-integrations";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
import * as cw_actions from "aws-cdk-lib/aws-cloudwatch-actions";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecs_patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as events from "aws-cdk-lib/aws-events";
import * as targets from "aws-cdk-lib/aws-events-targets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as kms from "aws-cdk-lib/aws-kms";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as lambda_event_sources from "aws-cdk-lib/aws-lambda-event-sources";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sns from "aws-cdk-lib/aws-sns";
import * as sqs from "aws-cdk-lib/aws-sqs";
import { Construct } from "constructs";
import { PlatformConfig } from "../config";

export interface ComputeStackProps extends cdk.StackProps {
  readonly cfg: PlatformConfig;
  readonly vpc: ec2.IVpc;
  readonly apiSecurityGroup: ec2.ISecurityGroup;
  readonly cacheSecurityGroup: ec2.ISecurityGroup;
  readonly sourceBucket: s3.IBucket;
  readonly multimodalBucket: s3.IBucket;
  readonly knowledgeBaseId: string;
  readonly dataSourceId: string;
  readonly redisEndpoint: string;
  readonly redisPort: string;
  readonly kmsKey: kms.IKey;
}

/**
 * ECS Fargate RAG API, HTTP API Gateway, SQS FIFO ingestion worker, and alarms.
 */
export class ComputeStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);
    const {
      cfg,
      vpc,
      apiSecurityGroup,
      sourceBucket,
      multimodalBucket,
      knowledgeBaseId,
      dataSourceId,
      redisEndpoint,
      redisPort,
      kmsKey,
    } = props;

    const alarmTopic = new sns.Topic(this, "Alarms", {
      displayName: `${cfg.prefix}-alarms`,
      masterKey: kmsKey,
    });

    const ingestDlq = new sqs.Queue(this, "IngestDlq", {
      fifo: true,
      queueName: `${cfg.prefix}-ingest-dlq.fifo`,
      encryption: sqs.QueueEncryption.KMS,
      encryptionMasterKey: kmsKey,
      retentionPeriod: cdk.Duration.days(14),
      contentBasedDeduplication: true,
    });

    const ingestQueue = new sqs.Queue(this, "IngestQueue", {
      fifo: true,
      queueName: `${cfg.prefix}-ingest.fifo`,
      encryption: sqs.QueueEncryption.KMS,
      encryptionMasterKey: kmsKey,
      visibilityTimeout: cdk.Duration.minutes(6),
      retentionPeriod: cdk.Duration.days(4),
      contentBasedDeduplication: true,
      deadLetterQueue: { queue: ingestDlq, maxReceiveCount: 5 },
    });

    const workerRole = new iam.Role(this, "IngestWorkerRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
    });
    workerRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaVPCAccessExecutionRole"),
    );
    workerRole.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          "bedrock:StartIngestionJob",
          "bedrock:ListIngestionJobs",
          "bedrock:GetIngestionJob",
        ],
        resources: ["*"],
      }),
    );
    sourceBucket.grantReadWrite(workerRole);
    kmsKey.grantEncryptDecrypt(workerRole);
    ingestQueue.grantConsumeMessages(workerRole);

    const worker = new lambda.Function(this, "IngestWorker", {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "workers.ingestion_handler.handler",
      code: lambda.Code.fromAsset(path.join(__dirname, "../../../src"), {
        bundling: {
          image: lambda.Runtime.PYTHON_3_12.bundlingImage,
          command: [
            "bash",
            "-c",
            "pip install -r workers/requirements.txt -t /asset-output && cp -R . /asset-output",
          ],
        },
      }),
      timeout: cdk.Duration.minutes(5),
      memorySize: 512,
      tracing: lambda.Tracing.ACTIVE,
      reservedConcurrentExecutions: cfg.environment === "prod" ? 20 : 5,
      environment: {
        AWS_REGION: this.region,
        KNOWLEDGE_BASE_ID: knowledgeBaseId,
        DATA_SOURCE_ID: dataSourceId,
        SOURCE_BUCKET: sourceBucket.bucketName,
        ENVIRONMENT: cfg.environment,
        SERVICE_NAME: "mmrag-ingest-worker",
      },
      role: workerRole,
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [apiSecurityGroup],
    });
    worker.addEventSource(
      new lambda_event_sources.SqsEventSource(ingestQueue, {
        batchSize: 10,
        reportBatchItemFailures: true,
        maxConcurrency: 4,
      }),
    );

    new events.Rule(this, "SourceObjectCreated", {
      eventPattern: {
        source: ["aws.s3"],
        detailType: ["Object Created"],
        detail: { bucket: { name: [sourceBucket.bucketName] } },
      },
      targets: [
        new targets.SqsQueue(ingestQueue, {
          // Per-object group: StartIngestionJob is KB-wide and de-duplicated in the worker.
          messageGroupId: events.EventField.fromPath("$.detail.object.key"),
        }),
      ],
    });

    const cluster = new ecs.Cluster(this, "Cluster", {
      vpc,
      containerInsights: true,
      clusterName: `${cfg.prefix}-cluster`,
    });

    const taskRole = new iam.Role(this, "ApiTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    taskRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "BedrockRagApis",
        actions: [
          "bedrock:Retrieve",
          "bedrock:RetrieveAndGenerate",
          "bedrock:RetrieveAndGenerateStream",
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:Converse",
          "bedrock:ConverseStream",
          "bedrock:StartIngestionJob",
          "bedrock:ListIngestionJobs",
        ],
        resources: ["*"],
      }),
    );
    taskRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "EmitMetricsAndTraces",
        actions: [
          "cloudwatch:PutMetricData",
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
        ],
        resources: ["*"],
      }),
    );
    sourceBucket.grantReadWrite(taskRole);
    multimodalBucket.grantRead(taskRole);
    kmsKey.grantEncryptDecrypt(taskRole);

    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(this, "Api", {
      cluster,
      cpu: 1024,
      memoryLimitMiB: 2048,
      desiredCount: cfg.desiredApiCount,
      publicLoadBalancer: true,
      circuitBreaker: { rollback: true },
      securityGroups: [apiSecurityGroup],
      taskSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      taskImageOptions: {
        image: ecs.ContainerImage.fromAsset(path.join(__dirname, "../../.."), {
          file: "Dockerfile",
        }),
        containerPort: 8080,
        taskRole,
        enableLogging: true,
        logDriver: ecs.LogDrivers.awsLogs({
          streamPrefix: "mmrag-api",
          logRetention: logs.RetentionDays.ONE_MONTH,
        }),
        environment: {
          AWS_REGION: this.region,
          ENVIRONMENT: cfg.environment,
          SERVICE_NAME: "mmrag-api",
          KNOWLEDGE_BASE_ID: knowledgeBaseId,
          DATA_SOURCE_ID: dataSourceId,
          SOURCE_BUCKET: sourceBucket.bucketName,
          MULTIMODAL_BUCKET: multimodalBucket.bucketName,
          REDIS_ENDPOINT: redisEndpoint,
          REDIS_PORT: redisPort,
          REDIS_TLS: "true",
          ENABLE_XRAY: "true",
          CLAUDE_SONNET_MODEL_ID: cfg.claudeSonnetModelId,
          CLAUDE_HAIKU_MODEL_ID: cfg.claudeHaikuModelId,
          BEDROCK_EMBEDDING_MODEL_ID: cfg.novaModelId,
          EMBEDDING_DIMENSION: String(cfg.embeddingDimension),
          CACHE_EMBEDDING_DIMENSION: "384",
        },
      },
      healthCheckGracePeriod: cdk.Duration.seconds(60),
    });
    service.targetGroup.configureHealthCheck({
      path: "/health",
      interval: cdk.Duration.seconds(30),
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 3,
    });
    service.targetGroup.setAttribute("deregistration_delay.timeout_seconds", "30");
    service.loadBalancer.setAttribute("idle_timeout.timeout_seconds", "120");

    const scaling = service.service.autoScaleTaskCount({
      minCapacity: cfg.desiredApiCount,
      maxCapacity: cfg.maxApiCount,
    });
    scaling.scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: 55,
      scaleInCooldown: cdk.Duration.seconds(60),
      scaleOutCooldown: cdk.Duration.seconds(30),
    });
    scaling.scaleOnRequestCount("RpsScaling", {
      requestsPerTarget: 80,
      targetGroup: service.targetGroup,
    });

    const httpApi = new apigwv2.HttpApi(this, "HttpApi", {
      apiName: `${cfg.prefix}-http`,
      description: "HTTP API facade for multimodal RAG (non-stream routes; stream prefers ALB)",
      corsPreflight: {
        allowHeaders: ["Authorization", "Content-Type", "X-Request-Id"],
        allowMethods: [
          apigwv2.CorsHttpMethod.GET,
          apigwv2.CorsHttpMethod.POST,
          apigwv2.CorsHttpMethod.OPTIONS,
        ],
        allowOrigins: ["*"],
      },
    });
    const albIntegration = new apigwv2i.HttpUrlIntegration(
      "AlbUrlIntegration",
      `http://${service.loadBalancer.loadBalancerDnsName}`,
    );
    httpApi.addRoutes({
      path: "/{proxy+}",
      methods: [apigwv2.HttpMethod.ANY],
      integration: albIntegration,
    });
    httpApi.addRoutes({
      path: "/health",
      methods: [apigwv2.HttpMethod.GET],
      integration: albIntegration,
    });

    const p99 = new cloudwatch.Alarm(this, "LatencyP99", {
      alarmDescription: "RAG API ALB p99 target response time > 3s",
      metric: service.targetGroup.metrics.targetResponseTime({
        statistic: "p99",
        period: cdk.Duration.minutes(1),
      }),
      threshold: 3,
      evaluationPeriods: 3,
      datapointsToAlarm: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    p99.addAlarmAction(new cw_actions.SnsAction(alarmTopic));

    const errorRate = new cloudwatch.Alarm(this, "5xxRate", {
      alarmDescription: "ALB 5XX count",
      metric: service.targetGroup.metrics.httpCodeTarget(elbv2.HttpCodeTarget.TARGET_5XX_COUNT, {
        period: cdk.Duration.minutes(1),
      }),
      threshold: 25,
      evaluationPeriods: 3,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    errorRate.addAlarmAction(new cw_actions.SnsAction(alarmTopic));

    const dlqAlarm = new cloudwatch.Alarm(this, "IngestDlqDepth", {
      metric: ingestDlq.metricApproximateNumberOfMessagesVisible(),
      threshold: 1,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    dlqAlarm.addAlarmAction(new cw_actions.SnsAction(alarmTopic));

    new cloudwatch.Alarm(this, "CacheMissSpike", {
      alarmDescription: "Cache miss count from EMF namespace MMRAG",
      metric: new cloudwatch.Metric({
        namespace: "MMRAG",
        metricName: "CacheMiss",
        statistic: "Sum",
        period: cdk.Duration.minutes(5),
      }),
      threshold: cfg.environment === "prod" ? 50_000 : 5_000,
      evaluationPeriods: 2,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    new cdk.CfnOutput(this, "AlbUrl", { value: `http://${service.loadBalancer.loadBalancerDnsName}` });
    new cdk.CfnOutput(this, "HttpApiUrl", { value: httpApi.apiEndpoint });
    new cdk.CfnOutput(this, "StreamHint", {
      value: `Use ALB URL for /v1/rag/stream (API Gateway HTTP APIs cap integrations at 30s).`,
    });
    new cdk.CfnOutput(this, "IngestQueueUrl", { value: ingestQueue.queueUrl });
  }
}

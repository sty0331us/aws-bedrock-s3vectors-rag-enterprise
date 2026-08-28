import * as cdk from "aws-cdk-lib";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as elasticache from "aws-cdk-lib/aws-elasticache";
import * as iam from "aws-cdk-lib/aws-iam";
import * as kms from "aws-cdk-lib/aws-kms";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as s3vectors from "aws-cdk-lib/aws-s3vectors";
import { Construct } from "constructs";
import { PlatformConfig } from "../config";

export interface DataStackProps extends cdk.StackProps {
  readonly cfg: PlatformConfig;
  readonly vpc: ec2.IVpc;
  readonly cacheSecurityGroup: ec2.ISecurityGroup;
}

/**
 * Source + multimodal S3, S3 Vectors bucket/index, Bedrock Knowledge Base,
 * and ElastiCache Serverless (L1/L2 cache).
 */
export class DataStack extends cdk.Stack {
  public readonly sourceBucket: s3.Bucket;
  public readonly multimodalBucket: s3.Bucket;
  public readonly knowledgeBase: bedrock.CfnKnowledgeBase;
  public readonly dataSource: bedrock.CfnDataSource;
  public readonly kmsKey: kms.Key;
  public readonly redisEndpoint: string;
  public readonly redisPort: string;

  constructor(scope: Construct, id: string, props: DataStackProps) {
    super(scope, id, props);
    const { cfg, vpc, cacheSecurityGroup } = props;

    this.kmsKey = new kms.Key(this, "DataKey", {
      enableKeyRotation: true,
      description: `${cfg.prefix} multimodal RAG data key`,
      alias: `${cfg.prefix}-data`,
    });

    const logsBucket = new s3.Bucket(this, "AccessLogs", {
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      lifecycleRules: [{ expiration: cdk.Duration.days(90) }],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.sourceBucket = new s3.Bucket(this, "SourceBucket", {
      bucketName: undefined,
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: this.kmsKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true,
      eventBridgeEnabled: true,
      serverAccessLogsBucket: logsBucket,
      serverAccessLogsPrefix: "source/",
      lifecycleRules: [
        {
          abortIncompleteMultipartUploadAfter: cdk.Duration.days(7),
          transitions: [
            { storageClass: s3.StorageClass.INTELLIGENT_TIERING, transitionAfter: cdk.Duration.days(30) },
          ],
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.multimodalBucket = new s3.Bucket(this, "MultimodalBucket", {
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: this.kmsKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: true,
      serverAccessLogsBucket: logsBucket,
      serverAccessLogsPrefix: "multimodal/",
      lifecycleRules: [
        {
          // Bedrock stores transient Nova MME artifacts under .bda/; expire them if GC misses.
          prefix: ".bda/",
          expiration: cdk.Duration.days(7),
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    const vectorBucket = new s3vectors.CfnVectorBucket(this, "VectorBucket", {
      vectorBucketName: `${cfg.prefix}-vectors-${this.account}`.slice(0, 63),
      encryptionConfiguration: {
        sseType: "aws:kms",
        kmsKeyArn: this.kmsKey.keyArn,
      },
    });
    vectorBucket.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN);

    const vectorIndex = new s3vectors.CfnIndex(this, "VectorIndex", {
      vectorBucketName: vectorBucket.vectorBucketName,
      indexName: "nova-mme-1024",
      dataType: "float32",
      dimension: cfg.embeddingDimension,
      distanceMetric: "cosine",
      metadataConfiguration: {
        // Required so Bedrock can store chunk text/metadata beyond filterable size limits.
        nonFilterableMetadataKeys: ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"],
      },
    });
    vectorIndex.addDependency(vectorBucket);
    vectorIndex.applyRemovalPolicy(cdk.RemovalPolicy.RETAIN);

    const kbRole = new iam.Role(this, "KnowledgeBaseRole", {
      assumedBy: new iam.ServicePrincipal("bedrock.amazonaws.com"),
      description: "Least-privilege Bedrock Knowledge Base execution role",
    });
    kbRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "InvokeNovaMultimodalEmbeddings",
        actions: ["bedrock:InvokeModel"],
        resources: [
          `arn:aws:bedrock:${this.region}::foundation-model/${cfg.novaModelId}`,
        ],
      }),
    );
    kbRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "ReadSourceObjects",
        actions: ["s3:GetObject", "s3:GetObjectVersion", "s3:ListBucket"],
        resources: [this.sourceBucket.bucketArn, this.sourceBucket.arnForObjects("*")],
      }),
    );
    kbRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "MultimodalSupplementalStorage",
        actions: [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ],
        resources: [this.multimodalBucket.bucketArn, this.multimodalBucket.arnForObjects("*")],
      }),
    );
    kbRole.addToPolicy(
      new iam.PolicyStatement({
        sid: "S3VectorIndexAccess",
        actions: [
          "s3vectors:PutVectors",
          "s3vectors:GetVectors",
          "s3vectors:DeleteVectors",
          "s3vectors:QueryVectors",
          "s3vectors:GetIndex",
          "s3vectors:ListVectors",
        ],
        resources: [vectorIndex.attrIndexArn],
      }),
    );
    this.kmsKey.grantEncryptDecrypt(kbRole);

    new s3vectors.CfnVectorBucketPolicy(this, "VectorBucketPolicy", {
      vectorBucketName: vectorBucket.vectorBucketName,
      policy: {
        Version: "2012-10-17",
        Statement: [
          {
            Sid: "AllowBedrockKnowledgeBase",
            Effect: "Allow",
            Principal: { AWS: kbRole.roleArn },
            Action: [
              "s3vectors:PutVectors",
              "s3vectors:GetVectors",
              "s3vectors:DeleteVectors",
              "s3vectors:QueryVectors",
              "s3vectors:GetIndex",
              "s3vectors:ListVectors",
              "s3vectors:GetVectorBucket",
            ],
            Resource: [
              vectorBucket.attrVectorBucketArn,
              `${vectorBucket.attrVectorBucketArn}/*`,
            ],
          },
        ],
      },
    });

    this.knowledgeBase = new bedrock.CfnKnowledgeBase(this, "KnowledgeBase", {
      name: `${cfg.prefix}-kb`,
      description: "Multimodal RAG knowledge base on S3 Vectors + Nova MME",
      roleArn: kbRole.roleArn,
      knowledgeBaseConfiguration: {
        type: "VECTOR",
        vectorKnowledgeBaseConfiguration: {
          embeddingModelArn: `arn:aws:bedrock:${this.region}::foundation-model/${cfg.novaModelId}`,
          embeddingModelConfiguration: {
            bedrockEmbeddingModelConfiguration: {
              dimensions: cfg.embeddingDimension,
              embeddingDataType: "FLOAT32",
              audio: [
                {
                  segmentationConfiguration: { fixedLengthDuration: cfg.audioVideoChunkSeconds },
                },
              ],
              video: [
                {
                  segmentationConfiguration: { fixedLengthDuration: cfg.audioVideoChunkSeconds },
                },
              ],
            },
          },
          supplementalDataStorageConfiguration: {
            storageLocations: [
              {
                type: "S3",
                s3Location: { uri: `s3://${this.multimodalBucket.bucketName}/` },
              },
            ],
          },
        },
      },
      storageConfiguration: {
        type: "S3_VECTORS",
        s3VectorsConfiguration: {
          indexArn: vectorIndex.attrIndexArn,
        },
      },
    });
    this.knowledgeBase.addDependency(vectorIndex);

    this.dataSource = new bedrock.CfnDataSource(this, "DataSource", {
      name: `${cfg.prefix}-s3-source`,
      knowledgeBaseId: this.knowledgeBase.attrKnowledgeBaseId,
      dataDeletionPolicy: "RETAIN",
      dataSourceConfiguration: {
        type: "S3",
        s3Configuration: {
          bucketArn: this.sourceBucket.bucketArn,
        },
      },
      vectorIngestionConfiguration: {
        chunkingConfiguration: {
          chunkingStrategy: "FIXED_SIZE",
          fixedSizeChunkingConfiguration: {
            maxTokens: 512,
            overlapPercentage: 20,
          },
        },
      },
    });

    const redis = new elasticache.CfnServerlessCache(this, "Redis", {
      engine: "redis",
      serverlessCacheName: `${cfg.prefix}-cache`.replace(/[^a-zA-Z0-9-]/g, "-").slice(0, 40),
      majorEngineVersion: "7",
      securityGroupIds: [cacheSecurityGroup.securityGroupId],
      subnetIds: vpc.selectSubnets({ subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS }).subnetIds,
      dailySnapshotTime: "05:00",
      snapshotRetentionLimit: cfg.environment === "prod" ? 7 : 1,
      kmsKeyId: this.kmsKey.keyId,
    });

    this.redisEndpoint = redis.attrEndpointAddress;
    this.redisPort = redis.attrEndpointPort;

    new cdk.CfnOutput(this, "SourceBucketName", { value: this.sourceBucket.bucketName });
    new cdk.CfnOutput(this, "MultimodalBucketName", { value: this.multimodalBucket.bucketName });
    new cdk.CfnOutput(this, "VectorBucketName", { value: vectorBucket.vectorBucketName ?? "" });
    new cdk.CfnOutput(this, "KnowledgeBaseId", { value: this.knowledgeBase.attrKnowledgeBaseId });
    new cdk.CfnOutput(this, "DataSourceId", { value: this.dataSource.attrDataSourceId });
    new cdk.CfnOutput(this, "RedisEndpoint", { value: this.redisEndpoint });
  }
}

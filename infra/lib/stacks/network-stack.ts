import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";
import { PlatformConfig } from "../config";

export interface NetworkStackProps extends cdk.StackProps {
  readonly cfg: PlatformConfig;
}

/**
 * Dual-AZ VPC with NAT for Fargate egress plus interface/gateway endpoints so
 * Bedrock, S3, SQS, ECR, and telemetry traffic does not traverse NAT at 100M MAU.
 */
export class NetworkStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly apiSecurityGroup: ec2.SecurityGroup;
  public readonly cacheSecurityGroup: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);
    const { cfg } = props;

    this.vpc = new ec2.Vpc(this, "Vpc", {
      vpcName: `${cfg.prefix}-vpc`,
      maxAzs: 2,
      natGateways: cfg.environment === "prod" ? 2 : 1,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: "private", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
        { name: "isolated", subnetType: ec2.SubnetType.PRIVATE_ISOLATED, cidrMask: 24 },
      ],
      gatewayEndpoints: {
        S3: { service: ec2.GatewayVpcEndpointAwsService.S3 },
      },
    });

    const endpointSg = new ec2.SecurityGroup(this, "VpceSg", {
      vpc: this.vpc,
      description: "HTTPS to VPC interface endpoints",
      allowAllOutbound: true,
    });
    endpointSg.addIngressRule(ec2.Peer.ipv4(this.vpc.vpcCidrBlock), ec2.Port.tcp(443));

    const interfaceEndpoints: Array<[string, ec2.IInterfaceVpcEndpointService]> = [
      ["EcrApi", ec2.InterfaceVpcEndpointAwsService.ECR],
      ["EcrDkr", ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ["Logs", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS],
      ["Monitoring", ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING],
      ["Sqs", ec2.InterfaceVpcEndpointAwsService.SQS],
      ["Secrets", ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER],
      ["BedrockRuntime", new ec2.InterfaceVpcEndpointAwsService("bedrock-runtime")],
      ["BedrockAgentRuntime", new ec2.InterfaceVpcEndpointAwsService("bedrock-agent-runtime")],
      ["BedrockAgent", new ec2.InterfaceVpcEndpointAwsService("bedrock-agent")],
      ["Bedrock", new ec2.InterfaceVpcEndpointAwsService("bedrock")],
      ["Xray", new ec2.InterfaceVpcEndpointAwsService("xray")],
    ];
    for (const [id, service] of interfaceEndpoints) {
      this.vpc.addInterfaceEndpoint(id, {
        service,
        securityGroups: [endpointSg],
        privateDnsEnabled: true,
        subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      });
    }

    this.apiSecurityGroup = new ec2.SecurityGroup(this, "ApiSg", {
      vpc: this.vpc,
      description: "Fargate RAG API tasks",
      allowAllOutbound: true,
    });
    this.apiSecurityGroup.addIngressRule(
      ec2.Peer.ipv4(this.vpc.vpcCidrBlock),
      ec2.Port.tcp(8080),
      "ALB and VPC link to API tasks",
    );

    this.cacheSecurityGroup = new ec2.SecurityGroup(this, "CacheSg", {
      vpc: this.vpc,
      description: "ElastiCache Serverless Redis",
      allowAllOutbound: false,
    });
    this.cacheSecurityGroup.addIngressRule(
      this.apiSecurityGroup,
      ec2.Port.tcp(6379),
      "API tasks to Redis",
    );

    new cdk.CfnOutput(this, "VpcId", { value: this.vpc.vpcId });
  }
}

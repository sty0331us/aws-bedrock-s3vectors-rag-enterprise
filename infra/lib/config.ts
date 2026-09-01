import * as cdk from "aws-cdk-lib";

export interface PlatformConfig {
  readonly project: string;
  readonly environment: string;
  readonly region: string;
  readonly prefix: string;
  readonly embeddingDimension: number;
  readonly audioVideoChunkSeconds: number;
  readonly novaModelId: string;
  readonly claudeSonnetModelId: string;
  readonly claudeHaikuModelId: string;
  readonly desiredApiCount: number;
  readonly maxApiCount: number;
}

export function loadConfig(app: cdk.App): PlatformConfig {
  const environment = app.node.tryGetContext("environment") ?? "dev";
  const project = app.node.tryGetContext("project") ?? "mmrag";
  const region = app.node.tryGetContext("region") ?? process.env.CDK_DEFAULT_REGION ?? "us-east-1";
  return {
    project,
    environment,
    region,
    prefix: `${project}-${environment}`,
    embeddingDimension: 1024,
    audioVideoChunkSeconds: 5,
    novaModelId: "amazon.nova-2-multimodal-embeddings-v1:0",
    claudeSonnetModelId: "us.anthropic.claude-sonnet-5",
    claudeHaikuModelId: "us.anthropic.claude-haiku-5",
    desiredApiCount: environment === "prod" ? 8 : 2,
    maxApiCount: environment === "prod" ? 64 : 4,
  };
}

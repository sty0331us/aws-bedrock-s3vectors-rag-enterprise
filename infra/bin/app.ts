#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { loadConfig } from "../lib/config";
import { NetworkStack } from "../lib/stacks/network-stack";
import { DataStack } from "../lib/stacks/data-stack";
import { ComputeStack } from "../lib/stacks/compute-stack";

const app = new cdk.App();
const cfg = loadConfig(app);

const env: cdk.Environment = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: cfg.region,
};

const tags = {
  Project: cfg.project,
  Environment: cfg.environment,
  Workload: "multimodal-rag",
};

const network = new NetworkStack(app, `${cfg.prefix}-network`, { env, cfg });
const data = new DataStack(app, `${cfg.prefix}-data`, {
  env,
  cfg,
  vpc: network.vpc,
  cacheSecurityGroup: network.cacheSecurityGroup,
});
data.addDependency(network);

const compute = new ComputeStack(app, `${cfg.prefix}-compute`, {
  env,
  cfg,
  vpc: network.vpc,
  apiSecurityGroup: network.apiSecurityGroup,
  cacheSecurityGroup: network.cacheSecurityGroup,
  sourceBucket: data.sourceBucket,
  multimodalBucket: data.multimodalBucket,
  knowledgeBaseId: data.knowledgeBase.attrKnowledgeBaseId,
  dataSourceId: data.dataSource.attrDataSourceId,
  redisEndpoint: data.redisEndpoint,
  redisPort: data.redisPort,
  kmsKey: data.kmsKey,
});
compute.addDependency(data);

Object.entries(tags).forEach(([key, value]) => cdk.Tags.of(app).add(key, value));

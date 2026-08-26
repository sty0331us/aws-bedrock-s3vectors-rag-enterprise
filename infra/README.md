# CDK notes

```bash
npm install
npx cdk synth -c environment=dev
npx cdk deploy --all -c environment=prod -c region=us-east-1
```

Requires AWS CDK v2 with `aws-cdk-lib` that includes `aws_s3vectors` (`CfnVectorBucket`, `CfnIndex`, `CfnVectorBucketPolicy`) and Bedrock `CfnKnowledgeBase` `S3_VECTORS` storage.

Pin `aws-cdk-lib` ≥ 2.220. If synth fails on `aws-s3vectors`, upgrade the CDK libraries — the L1 constructs shipped with S3 Vectors GA.

"""Long-lived AWS clients. Creating an aiobotocore client per request is too expensive at 100M MAU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiobotocore.session import AioSession


@dataclass(slots=True)
class AwsClients:
    bedrock_runtime: Any
    bedrock_agent_runtime: Any
    bedrock_agent: Any
    s3: Any

    @classmethod
    async def connect(cls, session: AioSession, region: str) -> AwsClients:
        runtime = await session.create_client("bedrock-runtime", region_name=region).__aenter__()
        agent_runtime = await session.create_client(
            "bedrock-agent-runtime", region_name=region
        ).__aenter__()
        agent = await session.create_client("bedrock-agent", region_name=region).__aenter__()
        s3 = await session.create_client("s3", region_name=region).__aenter__()
        return cls(
            bedrock_runtime=runtime,
            bedrock_agent_runtime=agent_runtime,
            bedrock_agent=agent,
            s3=s3,
        )

    async def close(self) -> None:
        for client in (
            self.bedrock_runtime,
            self.bedrock_agent_runtime,
            self.bedrock_agent,
            self.s3,
        ):
            await client.__aexit__(None, None, None)

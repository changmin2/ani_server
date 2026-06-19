import json
import os
from urllib.parse import parse_qs, urlparse, urlunparse

from azure.ai.projects import AIProjectClient
from azure.core.exceptions import AzureError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI


load_dotenv()


validation_input = {
    "documentType": "금융상품 안내문",
    "targetLanguage": "English",
    "country": "Vietnam",
    "sourceText": "우대금리는 최대 연 0.50%p 제공됩니다.",
    "translatedText": "Special interest rate available.",
    "keyInformation": [
        {
            "label": "우대금리",
            "sourceValue": "최대 연 0.50%p",
        }
    ],
}


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def split_endpoint_and_api_version(endpoint):
    parsed = urlparse(endpoint)
    query = parse_qs(parsed.query)
    api_versions = query.get("api-version", [])
    clean_endpoint = urlunparse(parsed._replace(query=""))

    return clean_endpoint.rstrip("/"), api_versions[0] if api_versions else None


def get_project_endpoint(endpoint):
    endpoint = endpoint.rstrip("/")

    if "/agents/" in endpoint:
        return endpoint.split("/agents/", 1)[0]

    return endpoint


def get_agent_name(endpoint):
    endpoint = endpoint.rstrip("/")

    if "/agents/" not in endpoint:
        return os.getenv("FOUNDRY_VALIDATION_AGENT_NAME")

    return endpoint.split("/agents/", 1)[1].split("/", 1)[0]


def strip_responses_path(endpoint):
    endpoint = endpoint.rstrip("/")

    if endpoint.endswith("/responses"):
        return endpoint[: -len("/responses")]

    return endpoint


def get_service_root(endpoint):
    parsed = urlparse(endpoint)
    return f"{parsed.scheme}://{parsed.netloc}"


def get_model_base_url(endpoint):
    endpoint = strip_responses_path(endpoint)

    if endpoint.rstrip("/").endswith("/openai/v1"):
        return endpoint.rstrip("/")

    if "/agents/" in endpoint:
        return f"{get_service_root(endpoint)}/openai/v1"

    if "/api/projects/" in endpoint:
        return f"{get_service_root(endpoint)}/openai/v1"

    return f"{endpoint.rstrip('/')}/openai/v1"


def get_agent_base_url(endpoint):
    endpoint = strip_responses_path(endpoint)

    if "/agents/" in endpoint:
        return endpoint.rstrip("/")

    return None


def get_foundry_api_key():
    auth_mode = os.getenv("FOUNDRY_AUTH_MODE", "api_key").strip().lower()

    if auth_mode in {"azure_identity", "default_credential", "aad"}:
        return get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://ai.azure.com/.default",
        )

    return get_required_env("FOUNDRY_API_KEY")


def get_agent_api_version():
    return os.getenv("FOUNDRY_AGENT_API_VERSION") or get_required_env("FOUNDRY_API_VERSION")


def get_foundry_client(base_url, include_api_version=False):
    kwargs = {
        "api_key": get_foundry_api_key(),
        "base_url": base_url,
    }

    if include_api_version:
        kwargs["default_query"] = {
            "api-version": get_agent_api_version(),
        }

    return OpenAI(**kwargs)


def get_foundry_model_client():
    endpoint, _endpoint_api_version = split_endpoint_and_api_version(
        get_required_env("FOUNDRY_PROJECT_ENDPOINT")
    )

    base_url = os.getenv("FOUNDRY_OPENAI_ENDPOINT") or get_model_base_url(endpoint)

    return get_foundry_client(base_url.rstrip("/"))


def build_validation_prompt(payload):
    return f"""
너는 금융 문서 번역 검수 Agent다.
입력된 원문과 번역문을 비교하여 금융 핵심 정보가 누락되었는지 검수한다.

검수 기준:
- sourceText에 있는 핵심 수치, 금리, 기간, 한도, 수수료, 법적/주의 문구가 translatedText에 보존되었는지 확인한다.
- keyInformation에 있는 label/sourceValue는 반드시 번역문에 의미가 반영되어야 한다.
- targetLanguage와 country를 고려해 자연스러운 표현인지 확인한다.
- 원문에 없는 내용을 새로 만들어냈는지 확인한다.
- 반드시 JSON 객체만 응답한다. markdown code block은 쓰지 않는다.

입력 JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}

반환 형식:
{{
  "is_valid": false,
  "score": 0.0,
  "summary": "",
  "issues": [
    {{
      "type": "missing_key_information",
      "severity": "high",
      "label": "우대금리",
      "source_value": "최대 연 0.50%p",
      "message": ""
    }}
  ],
  "recommended_translation": ""
}}
"""


def call_gpt_41_model(payload):
    client = get_foundry_model_client()
    deployment_name = os.getenv("FOUNDRY_DEPLOYMENT_NAME", "gpt-4.1")
    prompt = build_validation_prompt(payload)

    try:
        response = client.responses.create(
            model=deployment_name,
            input=[
                {
                    "role": "system",
                    "content": "You validate financial translations and return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    except BadRequestError as e:
        message = str(e)

        if "api version not supported" in message.lower():
            raise RuntimeError(
                "현재 endpoint가 api-version 방식을 지원하지 않습니다. "
                "Foundry 프로젝트 엔드포인트는 보통 endpoint 뒤에 /openai/v3을 붙인 "
                "OpenAI-compatible 경로로 호출해야 합니다. 스크립트는 이 경로를 자동으로 "
                "사용하도록 수정되어 있으니 다시 실행해보세요."
            ) from e

        if "responses" not in message.lower() and "response" not in message.lower():
            raise

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You validate financial translations and return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )
    except TypeError:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": "You validate financial translations and return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

    content = extract_response_text(response)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content}


def extract_response_text(response):
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text.strip()

    if hasattr(response, "choices"):
        return response.choices[0].message.content.strip()

    output = getattr(response, "output", None)

    if output:
        text_parts = []

        for item in output:
            content = getattr(item, "content", None)

            if not content:
                continue

            for content_item in content:
                text = getattr(content_item, "text", None)

                if text:
                    text_parts.append(text)

        if text_parts:
            return "\n".join(text_parts).strip()

    return str(response)


def call_validation_agent(payload):
    agent_id = os.getenv("FOUNDRY_VALIDATION_AGENT_ID")
    endpoint, _endpoint_api_version = split_endpoint_and_api_version(
        get_required_env("FOUNDRY_PROJECT_ENDPOINT")
    )
    agent_base_url = get_agent_base_url(endpoint)
    agent_name = get_agent_name(endpoint)

    if not agent_id and not agent_base_url:
        return {
            "skipped": True,
            "reason": "FOUNDRY_VALIDATION_AGENT_ID or an Agent OpenAI endpoint is not set in .env",
        }

    if not agent_base_url:
        return call_validation_agent_with_project_client(payload, agent_name, agent_id)

    prompt = build_validation_prompt(payload)

    try:
        client = get_foundry_client(agent_base_url, include_api_version=True)
        response = client.responses.create(
            model=os.getenv("FOUNDRY_DEPLOYMENT_NAME", "gpt-4.1"),
            input=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            extra_query={
                "api-version": get_agent_api_version(),
            },
        )
    except BadRequestError as e:
        message = str(e)

        if "api version not supported" not in message.lower():
            raise

        return call_validation_agent_with_project_client(payload, agent_name, agent_id)

    content = extract_response_text(response)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content}


def call_validation_agent_with_project_client(payload, agent_name, agent_id=None):
    if not agent_name:
        return {
            "skipped": True,
            "agent_id": agent_id,
            "reason": "Agent endpoint api-version is not supported, and agent_name could not be inferred.",
            "input": payload,
        }

    endpoint, _endpoint_api_version = split_endpoint_and_api_version(
        get_required_env("FOUNDRY_PROJECT_ENDPOINT")
    )
    project_endpoint = get_project_endpoint(endpoint)
    prompt = build_validation_prompt(payload)

    try:
        project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),
            allow_preview=True,
        )
        client = project_client.get_openai_client(agent_name=agent_name)
        response = client.responses.create(
            input=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    except AzureError as e:
        return {
            "skipped": True,
            "agent_name": agent_name,
            "agent_id": agent_id,
            "reason": f"Project client Agent call failed: {e}",
            "input": payload,
        }

    content = extract_response_text(response)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content}


def main():
    print("=== 1차: gpt-4.1 모델 호출 검수 ===")
    model_result = call_gpt_41_model(validation_input)
    print(json.dumps(model_result, ensure_ascii=False, indent=2))

    print("\n=== 2차: aniValidation-agent Agent ID 기반 호출 ===")
    agent_result = call_validation_agent(validation_input)
    print(json.dumps(agent_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

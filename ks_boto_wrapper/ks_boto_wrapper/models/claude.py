from typing import Dict, Any, Optional, List
import boto3
import requests
from pydantic import BaseModel
from botocore.config import Config

url = "http://20.197.39.253:3000/api/model/awsboto"

# ----------------- Config ----------------- #
boto_config = Config(
    connect_timeout=5,
    read_timeout=5
)

# ----------------- Pydantic Models ----------------- #
class Message(BaseModel):
    role: str
    content: str

class ContentBlock(BaseModel):
    type: str
    text: str

class ClaudeResponseContent(BaseModel):
    type: str
    text: str

class ClaudeResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int

class ClaudeResponse(BaseModel):
    id: str
    type: str
    role: str
    model: str
    content: List[ContentBlock]
    stop_reason: str
    stop_sequence: Optional[str]
    usage: ClaudeResponseUsage

# ----------------- Claude Wrapper ----------------- #
class Claude:
    def __init__(self, client, model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        self.client = client
        self.model_id = model_id
        self.boto3_instance = boto3.client(
            self.client.runtime,
            region_name=self.client.region_name,
            aws_access_key_id=self.client.aws_access_key_id,
            aws_secret_access_key=self.client.aws_secret_access_key,
            endpoint_url=self.client.endpoint_url,
            config=boto_config
        )

    def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request and handle the response.
        """
        try:
            response = requests.request(method, url, headers=headers, json=json)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
    ) -> ClaudeResponse:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.aws_secret_access_key}"
        }

        payload = {
            "type": "generate_response",
            "data" : {
                "model": model,
                "messages": messages,
            }
        }

        try:
            response_data = self.make_request("POST", url, headers=headers, json=payload)
            return ClaudeResponse(**response_data)

        except Exception as e:
            raise RuntimeError(f"Claude API request failed: {e}")

    def create_guardrail(
        self,
        name: str,
        description: str,
        topicPolicyConfig: Dict[str, Any],
        contentPolicyConfig: Dict[str, Any],
        wordPolicyConfig: Dict[str, Any],
        sensitiveInformationPolicyConfig: Dict[str, Any],
        blockedInputMessaging: str,
        blockedOutputsMessaging: str
    ):
        """
        Sends a POST request to your guardrail API with the specified configuration.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.aws_secret_access_key}"  # Ensure this is secure
        }

        payload = {
            "type": "create_guardrail",
            "data": {
                "name": name,
                "description": description,
                "topicPolicyConfig": topicPolicyConfig,
                "contentPolicyConfig": contentPolicyConfig,
                "wordPolicyConfig": wordPolicyConfig,
                "sensitiveInformationPolicyConfig": sensitiveInformationPolicyConfig,
                "blockedInputMessaging": blockedInputMessaging,
                "blockedOutputsMessaging": blockedOutputsMessaging
            }
        }

        try:
            response = self.make_request("POST", url, headers=headers, json=payload)
            return response

        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}")
            return response

        except requests.RequestException as req_err:
            print(f"Request exception occurred: {req_err}")
            return response
    
    def apply_guardrail(
        self,
        guardrailIdentifier: str,
        guardrailVersion: str,
        source: str,
        content: str
    ):
        """
        Applies a guardrail to a given piece of content.

        Parameters:
            guardrailIdentifier (str): Unique ID or name of the guardrail.
            guardrailVersion (str): Specific version of the guardrail to apply.
            source (str): Source type (e.g., "chat", "upload", etc.).
            content (str): The content to evaluate.

        Returns:
            GuardrailResponse: Result of the guardrail application.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.aws_secret_access_key}" 
        }

        payload = {
            "type" : "apply_guardrail",
            "data" : {
                "guardrailIdentifier": guardrailIdentifier,
                "guardrailVersion": guardrailVersion,
                "source": source,
                "content": content
            }
        }

        try:
            response = self.make_request("POST", url, headers=headers, json=payload)
            return response

        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}")
            return response

        except requests.RequestException as req_err:
            print(f"Request exception occurred: {req_err}")
            return response

    def converse_stream(
        self,
        modelId: str,
        messages: List[Dict[str, str]],
        system: str,
        inferenceConfig: Dict[str, Any],
        additionalModelRequestFields: Dict[str, Any]
    ):
        """
        Streams a conversation response from a model using a custom inference endpoint.

        Parameters:
            modelId (str): The model identifier to use for inference.
            messages (List[Dict[str, str]]): List of chat messages, e.g. [{"role": "user", "content": "Hi"}].
            system (str): System prompt or instruction.
            inferenceConfig (Dict[str, Any]): Configuration like temperature, maxTokens, etc.
            additionalModelRequestFields (Dict[str, Any]): Any extra model-specific fields.

        Returns:
            GuardrailResponse: A streaming or collected response from the inference engine.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client.aws_secret_access_key}",  # Ensure this is handled securely
            "Accept": "text/event-stream"  # Needed for SSE or streaming responses
        }

        payload = { 
            "type" : "converse_stream",
            "data" : {
                "modelId": modelId,
                "messages": messages,
                "system": system,
                "inferenceConfig": inferenceConfig,
                "additionalModelRequestFields": additionalModelRequestFields
            }
        }

        try:
            response = self.make_request("POST", url, headers=headers, json=payload, stream=True)

            # streamed_output = ""
            # for line in response.iter_lines():
            #     if line:
            #         decoded_line = line.decode("utf-8")
            #         # optionally parse if the line contains SSE structure like: "data: {...}"
            #         if decoded_line.startswith("data: "):
            #             decoded_line = decoded_line[len("data: "):]
            #         streamed_output += decoded_line + "\n"
            #         print("Streaming:", decoded_line)

            return response

        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err.response.status_code} - {http_err.response.text}")
            return response

        except requests.RequestException as req_err:
            print(f"Request exception occurred: {req_err}")
            return response


    def generate_response(
        self,
        user_message: str,
        temperature: Optional[float] = 0.7,
        top_p: Optional[float] = 0.9,
        max_tokens: Optional[int] = 1024,
        model: Optional[str] = None
    ) -> ClaudeResponse:
        messages = [{"role": "user", "content": user_message}]
        return self.create(
            messages=messages,
            model=model or self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

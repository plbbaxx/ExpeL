from typing import Callable, List
import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import openai


class VLLMTokenizer:
    """Use the serving vLLM process as the tokenizer authority for Qwen.

    The official ExpeL environment pins an older Transformers release which
    cannot deserialize Qwen3's tokenizer.json.  vLLM has already loaded the
    same local Qwen checkpoint for generation, so its supported /tokenize API
    provides exact token IDs without loading a second model or changing the
    environment.
    """
    def __init__(self, api_base: str, model_name: str):
        if not api_base:
            raise ValueError('EXPEL_OPENAI_API_BASE is required for a non-GPT tokenizer')
        self.endpoint = api_base.rstrip('/')
        if self.endpoint.endswith('/v1'):
            self.endpoint = self.endpoint[:-3]
        self.endpoint += '/tokenize'
        self.model_name = model_name

    def encode(self, text: str, add_special_tokens: bool = False):
        payload = json.dumps({
            'model': self.model_name,
            'prompt': text,
            'add_special_tokens': add_special_tokens,
        }).encode('utf-8')
        request = Request(self.endpoint, data=payload, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=15) as response:
                decoded = json.loads(response.read().decode('utf-8'))
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f'vLLM tokenizer endpoint failed: {self.endpoint}') from error
        tokens = decoded.get('tokens')
        if not isinstance(tokens, list):
            raise RuntimeError(f'Unexpected vLLM tokenizer response: {decoded}')
        return tokens


class GPTWrapper:
    def __init__(self, llm_name: str, openai_api_key: str, long_ver: bool):
        self.model_name = llm_name
        if long_ver:
            llm_name = os.environ.get('EXPEL_LONG_CONTEXT_MODEL', llm_name)
        api_base = os.environ.get('EXPEL_OPENAI_API_BASE')
        temperature = float(os.environ.get('EXPEL_TEMPERATURE', '0'))
        top_p = float(os.environ.get('EXPEL_TOP_P', '1'))
        max_tokens = int(os.environ.get('EXPEL_MAX_TOKENS', '512'))
        seed = int(os.environ.get('EXPEL_SEED', '42'))
        self.llm = ChatOpenAI(
            model=llm_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
            openai_api_base=api_base,
            model_kwargs={'seed': seed},
        )
        self.tokenizer = None
        if 'gpt' not in llm_name.lower():
            self.tokenizer = VLLMTokenizer(api_base, llm_name)
        self.usage_log = os.environ.get('EXPEL_LLM_USAGE_LOG')

    def __call__(self, messages: List[ChatMessage], stop: List[str] = [], replace_newline: bool = True) -> str:
        kwargs = {}
        if stop != []:
            kwargs['stop'] = stop
        for i in range(6):
            try:
                started = time.time()
                result = self.llm.generate([messages], **kwargs)
                output = result.generations[0][0].text.strip('\n').strip()
                if self.usage_log:
                    usage = (result.llm_output or {}).get('token_usage', {})
                    record = {'model': self.model_name, 'elapsed_seconds': time.time() - started,
                              'prompt_tokens': usage.get('prompt_tokens'),
                              'completion_tokens': usage.get('completion_tokens'),
                              'total_tokens': usage.get('total_tokens')}
                    os.makedirs(os.path.dirname(self.usage_log), exist_ok=True)
                    with open(self.usage_log, 'a', encoding='utf-8') as handle:
                        handle.write(json.dumps(record) + '\n')
                break
            except openai.error.RateLimitError:
                print(f'\nRetrying {i}...')
                time.sleep(1)
        else:
            raise RuntimeError('Failed to generate response')

        if replace_newline:
            output = output.replace('\n', '')
        return output

def LLM_CLS(llm_name: str, openai_api_key: str, long_ver: bool) -> Callable:
    if 'gpt' in llm_name.lower() or os.environ.get('EXPEL_OPENAI_API_BASE'):
        return GPTWrapper(llm_name, openai_api_key, long_ver)
    else:
        raise ValueError(f"Unknown LLM model name: {llm_name}")

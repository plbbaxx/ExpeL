from typing import Callable, List
import json
import os
import re
import time

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import openai


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

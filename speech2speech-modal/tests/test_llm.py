import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import app
from llm import LLMService

@app.local_entrypoint()
def main():
    llm = LLMService()
    print(llm.generate.remote("Say hello in one sentence."))

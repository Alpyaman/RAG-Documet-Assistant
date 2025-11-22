# Phase 2: LLM Generation
 
This directory contains examples for Phase 2 of the RAG Document Assistant, which adds LLM-powered answer generation to the retrieval pipeline.
 
## What's New in Phase 2?
 
Phase 2 transforms the pipeline from simple retrieval to intelligent answer generation:
 
**Before (Phase 1):**
```
User: "What is the conclusion?"
System: [Returns 3 chunks of text]
```
 
**After (Phase 2):**
```
User: "What is the conclusion?"
System: "The conclusion is that X, Y, and Z based on the analysis presented in the document..."
```
 
## Architecture
 
### New Components
 
1. **`generator.py`** - LLM generation module
   - Abstract base class `BaseLLMGenerator`
   - `OpenAIGenerator` - Uses GPT models (paid, fast)
   - `OllamaGenerator` - Uses local models via Ollama (free, private)
   - `HuggingFaceGenerator` - Uses HuggingFace transformers (free, local)
 
2. **Updated `pipeline.py`**
   - New `generate_answer()` method
   - Combines retrieval + generation
   - Returns `GenerationResult` with LLM answer
 
3. **Updated `config.py`**
   - LLM provider settings
   - Model selection
   - API keys and endpoints
 
## Quick Start
 
### Option 1: OpenAI (Paid, Best Quality)
 
```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
 
# Run with OpenAI
python examples/phase2_generation_example.py --provider openai --model gpt-3.5-turbo
```
 
**Pros:**
- High quality answers
- Fast inference
- No local resources needed
 
**Cons:**
- Costs money per request
- Requires internet connection
- Data sent to OpenAI servers
 
### Option 2: Ollama (Free, Local)
 
```bash
# Install Ollama
# Visit: https://ollama.ai
 
# Pull a model
ollama pull llama2
 
# Run with Ollama
python examples/phase2_generation_example.py --provider ollama --model llama2
```
 
**Pros:**
- Completely free
- Private (runs locally)
- No API keys needed
 
**Cons:**
- Requires local GPU/CPU resources
- Slower inference (especially on CPU)
- Need to download models (GBs of data)
 
### Option 3: HuggingFace (Free, Local)
 
```bash
# Install dependencies
pip install transformers torch
 
# Run with HuggingFace
python examples/phase2_generation_example.py --provider huggingface --model google/flan-t5-base
```
 
**Pros:**
- Free and open source
- Runs locally
- Many models available
 
**Cons:**
- Requires torch and transformers
- Slower than OpenAI
- Quality varies by model
 
## Usage Examples
 
### Basic Usage
 
```python
from rag_assistant.pipeline import RAGPipeline
 
# Initialize pipeline (uses config defaults)
pipeline = RAGPipeline()
 
# Generate an answer
result = pipeline.generate_answer("What is the main topic?")
print(result.answer)
```
 
### With Custom Configuration
 
```python
from rag_assistant.pipeline import RAGPipeline
from rag_assistant.config import RagConfig
 
# Configure for Ollama
config = RagConfig(
    llm_provider="ollama",
    llm_model_name="llama2",
    llm_temperature=0.7,
    llm_max_tokens=512
)
 
pipeline = RAGPipeline(config=config)
result = pipeline.generate_answer("Summarize the document")
print(result.answer)
```
 
### Environment Variables
 
You can also configure via environment variables (`.env` file):
 
```bash
# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=512
 
# OpenAI
OPENAI_API_KEY=your-key-here
 
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```
 
## Comparison: Retrieval vs Generation
 
### Phase 1 (Retrieval Only)
```python
results = pipeline.query("What is the conclusion?", top_k=3)
for result in results:
    print(result['text'][:200])
```
 
**Output:**
```
[1] In this study, we examined the effects of...
[2] The data shows a significant correlation...
[3] Based on our analysis, we can conclude...
```
 
### Phase 2 (Retrieval + Generation)
```python
result = pipeline.generate_answer("What is the conclusion?")
print(result.answer)
```
 
**Output:**
```
The conclusion is that there is a significant correlation between X and Y,
as demonstrated by the study's findings. The analysis suggests that further
research is needed to explore Z. The authors recommend implementing policy
changes to address these issues. [1][2][3]
```
 
## Advanced Features
 
### Custom Prompting
 
You can subclass `BaseLLMGenerator` to customize prompts:
 
```python
from rag_assistant.generator import OpenAIGenerator
 
class CustomGenerator(OpenAIGenerator):
    def build_prompt(self, query: str, context: List[str]) -> str:
        # Custom prompt template
        return f"""
        Context: {context}
 
        Question: {query}
 
        Answer in bullet points:
        """
 
pipeline = RAGPipeline(generator=CustomGenerator())
```
 
### Accessing Context
 
```python
# Include context in the result
result = pipeline.generate_answer(
    "What is the conclusion?",
    return_context=True
)
 
print(result.answer)
print(f"\nBased on {len(result.context_used)} chunks:")
for i, chunk in enumerate(result.context_used, 1):
    print(f"  [{i}] {chunk[:100]}...")
```
 
## Cost Optimization
 
### Using Local Models
 
For production systems with high query volumes, consider:
 
1. **Ollama with quantized models** - Fast local inference
2. **HuggingFace with smaller models** - Balance quality/speed
3. **Hybrid approach** - Local for simple queries, OpenAI for complex ones
 
### Example: Hybrid Approach
 
```python
def smart_generate(query: str, pipeline: RAGPipeline):
    # Use local model for simple queries
    if len(query.split()) < 10:
        local_generator = create_generator("ollama", model_name="llama2")
        return local_generator.generate(query, context)
 
    # Use OpenAI for complex queries
    else:
        return pipeline.generate_answer(query)
```
 
## Troubleshooting
 
### OpenAI Issues
 
**Problem:** `OpenAI API key not found`
```bash
export OPENAI_API_KEY="your-key-here"
```
 
**Problem:** `Rate limit exceeded`
- Upgrade your OpenAI plan
- Add retry logic with exponential backoff
- Switch to local model
 
### Ollama Issues
 
**Problem:** `Cannot connect to Ollama`
```bash
# Check if Ollama is running
ollama list
 
# Start Ollama (if not running)
ollama serve
```
 
**Problem:** `Model not found`
```bash
# Pull the model first
ollama pull llama2
ollama pull mistral
```
 
### HuggingFace Issues
 
**Problem:** `transformers not installed`
```bash
pip install transformers torch
```
 
**Problem:** `CUDA out of memory`
```python
# Use CPU instead
config = RagConfig(
    llm_provider="huggingface",
    device="cpu"  # Use CPU instead of GPU
)
```
 
## Performance Benchmarks
 
Approximate response times (on standard hardware):
 
| Provider | Model | Speed | Quality | Cost |
|----------|-------|-------|---------|------|
| OpenAI | gpt-3.5-turbo | ~1-2s | Excellent | $$ |
| OpenAI | gpt-4 | ~3-5s | Best | $$$$ |
| Ollama | llama2 (7B) | ~5-10s (CPU) | Good | Free |
| Ollama | mistral (7B) | ~5-10s (CPU) | Very Good | Free |
| HuggingFace | flan-t5-base | ~3-5s (CPU) | Moderate | Free |
| HuggingFace | flan-t5-large | ~8-12s (CPU) | Good | Free |
 
## Next Steps
 
1. **Tune prompts** - Customize for your domain
2. **Add streaming** - Show answers as they generate
3. **Implement caching** - Cache answers for common questions
4. **Add feedback** - Collect user feedback to improve prompts
5. **Monitor quality** - Track answer quality metrics
 
## Resources
 
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
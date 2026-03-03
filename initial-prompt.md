I want to create some kind of an LLM router. Perhaps I just need to use an already existing one.

I have some custom code in ~/containers/llm/agent-service.py that adds a tool-calling proxy in front of vLLM. I want to include that functionality, but only for some LLMs.

I have 3 machines that can run LLMs with differing abilities. 2 machines are nVidia Spark machines, 1 is a Strix Halo AMD machine configured with 64 gig of video memory.

I want to have a library of local LLMs that can be used through the router. Some will be running on a vLLM instance. Perhaps others will be running in LM Studio? There will text, vision, and audio models.

I want there to be a set of LLMs that are always running. The others are on-demand. Some models will fit on a single node. Others will have to run across the 2 Spark machines. I want a single OpenAI-type URL that I can point all applications at.

I'd also like to have model aliases. For example, large-research, medium-coding, etc.  
I haven't looked too deeply, but I would be happy to use an existing solution out there, or extend one. Start by researching that first


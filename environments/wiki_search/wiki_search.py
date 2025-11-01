import os
from typing import cast, List, Dict, Any

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

def normalize_id(text: str) -> str:
    """Normalize text to create section IDs."""
    return text.strip().lower().replace(" ", "_")


class WikiSemanticSearchEnv(vf.SemanticSearchEnv):
    def __init__(
        self,
        corpus_dataset: str = "willcb/rare-wiki-pages",
        corpus_split: str = "train",
        collection_name: str = "wiki_titles",
        **kwargs
    ):
        self.corpus_dataset = corpus_dataset
        self.corpus_split = corpus_split
        self.page_id_to_title: Dict[str, str] = {}
        self.page_id_to_content: Dict[str, str] = {}
        
        super().__init__(collection_name=collection_name, **kwargs)
    
    def prep_corpus(self) -> None:
        corpus = load_dataset(self.corpus_dataset, split=self.corpus_split)
        
        page_ids = []
        titles = []
        metadatas = []
        
        for row in corpus:
            row = cast(dict, row)
            page_id = row["id"]
            title = row["title"]
            content = row["content"]
            
            self.page_id_to_title[page_id] = title
            self.page_id_to_content[page_id] = content
            
            page_ids.append(page_id)
            titles.append(title.strip())  # Use title as embedding text
            metadatas.append({
                "title": title,
            })
        
        self.upsert_documents(
            document_ids=page_ids,
            documents=titles,
            metadatas=metadatas,
            batch_size=500
        )

    async def search_pages(self, query: str) -> List[Dict]:
        """Search for relevant Wikipedia pages.
        
        Args:
            query: Search query
            
        Returns:
            List of dicts with page_id and title
            
        Example:
            "basketball" -> [{"page_id": "basketball", "title": "Basketball"}, ...]
        """
        results = await self.search_documents(
            query=query,
            n_results=10,
            return_contents=False,
            return_metadata=True
        )
        
        # wiki search format
        output = []
        for result in results:
            output.append({
                "page_id": result["document_id"],
                "title": result["metadata"]["title"] if result["metadata"] else "Unknown"
            })
        
        return output
    
    async def view_sections(self, page_id: str) -> List[Dict]:
        """View sections of a Wikipedia page.
        
        Args:
            page_id: The page ID
            
        Returns:
            List of dicts with section_id and section_name
            
        Example:
            "basketball" -> [{"section_id": "basketball:history", "section_name": "History"}, ...]
        """
        if page_id not in self.page_id_to_content:
            raise ValueError(f"Page not found: {page_id}")
        
        content = self.page_id_to_content[page_id]
        sections = []
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                section_id = f"{page_id}:{normalize_id(section_name)}"
                sections.append({
                    "section_id": section_id,
                    "section_name": section_name
                })
        
        # If no sections, return whole page
        if not sections:
            sections.append({
                "section_id": f"{page_id}:full",
                "section_name": "Full Page"
            })
        
        return sections
    
    async def read_section(self, section_id: str) -> str:
        """Read content of a specific section.
        
        Args:
            section_id: Section ID (format: "page_id:section_name")
            
        Returns:
            Content of the section
            
        Example:
            "basketball:history" -> "Basketball was invented in 1891..."
        """
        if ":" not in section_id:
            raise ValueError("Invalid section_id format. Expected: page_id:section_name")
        
        page_id, section_name_id = section_id.split(":", 1)
        
        if page_id not in self.page_id_to_content:
            raise ValueError(f"Page not found: {page_id}")
        
        content = self.page_id_to_content[page_id]
        lines = content.split("\n")
        
        if section_name_id == "full":
            return content
        
        section_start = None
        section_end = None
        
        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = normalize_id(line.lstrip("#").strip())
                if current_section == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break
        
        if section_start is not None:
            if section_end is None:
                section_end = len(lines)
            return "\n".join(lines[section_start:section_end])
        else:
            raise ValueError(f"Section not found: {section_id}")


async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
    judge_response = await judge(prompt, completion, answer, state)
    return 1.0 if "yes" in judge_response.lower() else 0.0


def load_environment(
    max_turns: int = 10,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs
) -> vf.Environment:
    dataset = load_dataset("willcb/wiki-trivia-questions", split="train")
    
    vf_env = WikiSemanticSearchEnv(
        dataset=dataset,
        max_turns=max_turns,
        **kwargs
    )

    # use variation of search document as the original env
    # searches and returns page titles instead of document contents
    # maybe we update this later to do more standard document search via content embd
    vf_env.remove_tool(vf_env.search_documents)
    vf_env.add_tool(vf_env.search_pages)
    vf_env.add_tool(vf_env.view_sections)
    vf_env.add_tool(vf_env.read_section)
    
    judge_client = OpenAI(
        base_url=judge_base_url,
        api_key=os.getenv(judge_api_key_var)
    )
    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=vf_env.parser
    )
    
    judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    vf_env.rubric = vf.RubricGroup(rubrics=[judge_rubric, vf_env.rubric])
    
    return vf_env

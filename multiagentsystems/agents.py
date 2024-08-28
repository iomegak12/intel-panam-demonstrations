from crewai import Agent
from crewai_tools import WebsiteSearchTool, SerperDevTool, FileReadTool

web_search_tool = WebsiteSearchTool()
serper_dev_tool = SerperDevTool()
file_read_tool = FileReadTool(
    file_path="./job_description_template.md",
    description="A tool to read the job description template file"
)


class MyAgents():
    def researcher_agent(self):
        return Agent(
            role="Research Agent",
            goal="Analyze the company website and provided description to extract insights on culture, values and specific needs.",
            tools=[web_search_tool, serper_dev_tool],
            backstory="Expert in analyzing company cultures and identifying key values and needs from various sources including websites and brief descriptions",
            verbose=True
        )

    def writer_agent(self):
        return Agent(
            role="Job Description Writer",
            goal="Use insights from the Research analyst to create a detailed, engaging and awesome job posting.",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="Skilled Professional in writing / crafting a compelling job description that outstand or resonate with the company values and attract the right candidates",
            verbose=True
        )

    def reviewer_agent(self):
        return Agent(
            role="Review and Editing Specialist",
            goal="Review the Job Posting for clarity, engagement, grammatical accuracy and alignment with company values and refint it to ensure that its' perfect",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="A strict and meticulous editor with an eye for detail, ensure every piece of content is clear, engaging and grammatically perfect.",
            verbose=True
        )

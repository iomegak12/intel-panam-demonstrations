from dotenv import load_dotenv
from crewai import Crew
from tasks import MyTasks
from agents import MyAgents


def main():
    load_dotenv()

    tasks = MyTasks()
    agents = MyAgents()

    company_description = input("Whats the company description?\n")
    company_domain = input("What's the company URL?\n")
    hiring_needs = input("What are all the hiring needs?\n")
    specific_benefits = input(
        "What are all the specific benefits you offer?\n")

    researcher = agents.researcher_agent()
    writer = agents.writer_agent()
    reviewer = agents.reviewer_agent()

    research_company = tasks.research_company_culture_task(
        agent=researcher, company_description=company_description, company_domain=company_domain)

    industry_analysis = tasks.industry_analysis_task(
        agent=researcher, company_domain=company_domain)

    research_role_requirements = tasks.research_role_requirements_task(
        agent=researcher, hiring_needs=hiring_needs)

    draft_job_posting = tasks.draft_job_posting_task(
        agent=writer, company_description=company_description,
        hiring_needs=hiring_needs, specific_benefits=specific_benefits)

    review_edit_job_posting = tasks.review_and_edit_job_posting_task(
        agent=reviewer, hiring_needs=hiring_needs)

    job_posting_crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[
            research_company,
            industry_analysis,
            research_role_requirements,
            draft_job_posting,
            review_edit_job_posting
        ]
    )

    result = job_posting_crew.kickoff()

    print("Job Posting Creation is Completed Successfully ...")
    print("Job Description Posting \n")

    print(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error Occurred, Details : {error}")

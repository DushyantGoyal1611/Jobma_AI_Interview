# Question Generating Prompt
question_prompt = PromptTemplate(
    template="""
    You are an AI Interview Question Generator.
    Based on the following candidate profile, generate 1 technical and 1 behavioral interview question **per skill**.

    Candidate Profile:
    - Skills: {skills}
    - Total Experience: {experience}

    Rules:
    - keep questions short and clear.
    - Use a mix of basic and intermediate-level questions.
    - Don't repeat skills.
    - For soft skills like "Team Management", "Leadership", or "Communication", generate situational or behavioral questions.

    {format_instruction}
    """,
    input_variables= ['skills', 'experience'],
    partial_variables={'format_instruction': question_parser.get_format_instructions()}
)


question_chain = (
    {"skills": lambda x: x["Skills"], "experience": lambda x: x["Experience"]}
    | question_prompt
    | llm
    | question_parser
)

print("=====================================================================\n=====================================================================")
questions = question_chain.invoke(resume_result)
print("Generated Questions: ", questions)
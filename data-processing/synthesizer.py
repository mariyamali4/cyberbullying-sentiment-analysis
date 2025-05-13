from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
    expected_output_format="Ensure the output resembles a medical chatbot tasked with diagnosing a patient’s illness. " \
    "It should pose additional questions if the details are inadequate or provide a diagnosis when the input is sufficiently detailed.",
    input_format="Mimic the kind of queries or statements a patient might share with a medical chatbot when seeking a diagnosis.",
    task="The chatbot acts as a specialist in medical diagnosis, integrated with a patient scheduling system. It manages tasks in a sequence to ensure precise and effective appointment setting and diagnosis processing.",
    scenario="Non-medical patients describing symptoms to seek a diagnosis.",
)

"Ensure the output resembles a subreddit post, made by users of different age groups, ethnicities, gender, and language backgrounds. " \
"Users should be discussing a variety of topics, including but not limited to: personal experiences, opinions, and questions. " 
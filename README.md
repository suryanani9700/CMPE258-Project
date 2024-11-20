# Introduction
Radiology by nature generates a lot of text data, this makes it an ideal case for an LLM. Some examples of the data modalities generated are as follows:
Reports
Clinical Notes with associated Images

The motivation behind this project stems from the significant challenges and costs associated with medical image analysis, particularly in radiology. Currently,  (1) radiologists spend extensive hours reviewing and interpreting medical images before generating detailed reports, a process that can take several hours per image. With the average radiologist in California earning $59.99 per hour—34% above the national average—and working long hours, the cost of labor for medical imaging analysis becomes substantial. 

Additionally, (2) radiologists require years of specialized training, with a typical educational path taking at least seven years. This makes the field both time and resource intensive.

By leveraging Large Language Models (LLMs), this project aims to automate the process of generating radiology-level reports, significantly reducing both the time and costs involved. Integrating specialized LLMs for medical tasks has the potential:
To streamline clinical workflows
To lower operational costs, and alleviate the burden of note taking for radiologists, allowing them to focus on more complex cases and improving overall efficiency in the healthcare system.
Radiology is a very text intensive field with doctors producing multiple reports,notes as a result this is a natural fit for LLMs, this in turn provides an opportunity for the LLM to simplify the report from a user perspective

# Why do this?
The project addresses the pressing challenges in medical imaging, particularly in radiology, by using Large Language Models (LLMs) for automating radiology report generation. Here are the key reasons motivating this initiative:
Labor and cost efficiency: Radiologists are highly trained specialists who command significant salaries, especially in locations like California where wages exceed the national average by 34%. Moreover, with radiologists working extensive hours, the cumulative cost of labor for generating radiology reports is substantial. Automating this task with LLMs can reduce labor time and associated costs drastically.
Time intensive training: Becoming a radiologist requires a minimum of seven years of specialized training. By automating simpler tasks like report generation, the expertise of radiologists can be redirected toward more complex cases, alleviating some pressure on the healthcare workforce and improving the allocation of human resources.
Workflow optimization: Medical imaging is critical in healthcare delivery. So automating these workflows using LLMs trained on paired image-text datasets can optimize clinical workflows, reducing delays and enabling faster diagnosis and treatment.
Potential for technology integration: Advanced in LLMs, such as the proposed integration with radiology images highlights the potential to push boundaries in AI-assisted healthcare. These systems can be scaled to improve accessibility to radiology services in under-resourced regions where there is a shortage of radiologists.

The projects can be divided into two parts:
First we show how we can design the integration of an LLM in a clinical radiological workflow
Second how we can use existing LLMs (customizable for specific radiology diseases specifically for hard to detect cases as Cancer) and the other how we can enhance the understanding and the simplification of the report from a user perspective

# References

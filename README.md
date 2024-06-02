## Team members

Daniel Cerkoney and Anthony Young

## Background and Objectives

The aim of this project is to study factors contributing to the survival of cancer patients in the US.
To that end, we used Kaplan-Meier estimation and Cox proportional hazards regression to analyze all brain tumor cases in the surveillance, epidemiology, and end results ([SEER](https://seer.cancer.gov/)) database from 2000–2020.
Data cleaning and pre-processing and the train-test split were performed using [scikit-learn](https://scikit-learn.org/stable/), while the [lifelines](https://lifelines.readthedocs.io/en/stable/) library was used to fit the survival models.

## Results

[Here](reports/slides.pdf) are some slides describing the project.
Finally, here are some examples of predicted survival functions adjusted for year of diagnosis using the Kaplan-Meier estimator and Cox regression, respectively:

<p align="middle">
  <img src="https://github.com/dcerkoney/spring-2024-cancer-survival-known-success/assets/11780326/dc0b8baa-c8c5-439c-a213-5aab5e291bd3" width="405" /> 
  <img src="https://github.com/dcerkoney/spring-2024-cancer-survival-known-success/assets/11780326/7f8a0975-a747-4c8d-b8f5-76e487b3a768" width="405" />
</p>

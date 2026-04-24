# Interactive Sales & Profit Analysis Tool

## 1. Problem & User

This project develops an interactive data analysis tool to support decision-making in a retail business context.
The tool is designed for retail managers, business analysts, and decision-makers who need to evaluate sales performance, profitability, and discount strategies.

The key analytical question is:
**How can managers identify which regions, product categories, and discount strategies drive revenue while maintaining profitability?**

---

## 2. Data

Dataset: Superstore Sales Dataset (public dataset)
Source: https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv
Access Date: April 2026

Key variables:

* Order Date
* Region
* Segment
* Category / Sub-Category
* Sales
* Profit
* Discount
* Quantity

This dataset was selected because it represents a realistic retail business environment and allows meaningful financial and operational analysis.

---

## 3. Methods

The project follows a complete Python analytical workflow:

* Data loading using pandas (online dataset)
* Data cleaning and type conversion
* Feature engineering:

  * Year / Month extraction
  * Profit Margin calculation
  * Discount band classification
* Descriptive and comparative analysis
* Visualisation using Plotly
* Interactive dashboard built using Streamlit

---

## 4. Key Findings

* High sales do not necessarily lead to high profitability.
* Certain categories generate strong revenue but low profit margins.
* Regional performance varies significantly.
* Higher discount levels are generally associated with lower profitability.
* Technology products tend to have more stable profit margins.

---

## 5. Product Link / Demo

* Interactive App: [Insert Streamlit Link Here] e.g.: https://github.com/****/project_name/Interactive_Data_Analysis_Tool.py
* Demo Video (1–3 min): [Insert Video Link Here] e.g.: https://github.com/****/project_name/***.mp4

---

## 6. How to Run

1. Install dependencies:

   ```
   conda env create -f requirements.yml
   ```

2. Run the app:

   ```
   python Interactive_Data_Analysis_Tool.py
   ```

---

## 7. Repository Structure

```
- Interactive_Data_Analysis_Tool.py (main application)
- Interactive_Data_Analysis_Tool.ipynb (analysis workflow)
- README.md
- requirements.yml
- Demo Video
```

---

## 8. Limitations & Future Improvements

* The dataset represents a simplified retail scenario
* The analysis is descriptive rather than predictive
* No causal inference is established

Future improvements:

* Add forecasting models
* Include customer segmentation
* Improve UI/UX design
* Integrate real-time datasets

---

## 9. Notes

* Data is publicly available and used for educational purposes
* No sensitive or personal data is included

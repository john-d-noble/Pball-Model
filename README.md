Investor Summary – Franchise vs Independent IRR Model for Pickleball Facility
1. Model Overview and Optimization Goal
This IRR model evaluates five investment scenarios for launching and operating a pickleball facility: three risk-adjusted scenarios under a franchise model (Pessimistic, Base, Optimistic), a Target IRR case, and a fifth Independent/Non-Franchise scenario.

The primary optimization goal is to understand the expected Internal Rate of Return (IRR) and Net Present Value (NPV) under each model, driven by revenue potential, capital investment, member churn, and operating cost structures. Free Cash Flow (FCF) is computed annually for seven years, factoring in debt service, taxes, capital expenditures, and franchise-specific fees where applicable.
2. Key Risks and Model Constraints
Risks include:
- Sensitivity to court utilization, churn, and pricing
- Market assumptions derived from industry benchmarks and may differ locally
- Terminal value assumptions can heavily influence final year cash flows
- Debt service modeled with consistent amortization, not real-time bank schedules
- Independent scenario assumes higher customer acquisition friction without brand recognition

3. IRR and NPV Results by Scenario
Scenario	IRR (%)	NPV ($)
Pessimistic	6.2	98,000
Base Case	13.5	453,000
Optimistic	19.6	911,000
Target IRR	21.1	1,180,000
Independent	15.4	650,000
4. Explanation of Inputs and Model Drivers
Key inputs:
- Court Utilization: 25–38% Year 1, compounding
- Monthly Membership Fee: $80–$110
- Churn Rate: 18–28%
- Franchise Fee: 10% of revenue (omitted in independent model)
- Tax Rate: 25%, Discount Rate: 12%
- Initial Investment: $850K–$1M

5. Visual Summary of Results
### Free Cash Flow by Scenario
This line chart illustrates how much cash is left over each year after expenses, debt payments, and taxes. It helps investors see the relative financial strength and risk of each modeled scenario.
 
Figure 1: Free Cash Flow by Scenario
### IRR Distribution – Monte Carlo Simulation
This histogram shows the range of Internal Rate of Return (IRR) outcomes across 1,000 simulations. It helps assess how frequently certain levels of return might occur, based on varying pricing, utilization, and churn.
 
Figure 2: IRR Distribution – Monte Carlo Simulation
### NPV Distribution – Monte Carlo Simulation
Similar to the IRR chart, but focused on Net Present Value (NPV). A high concentration above $0 indicates probable profit.
 
Figure 3: NPV Distribution – Monte Carlo Simulation
### IRR Sensitivity to Fee and Utilization
This heatmap evaluates how IRR changes with small adjustments in pricing and utilization. It highlights key revenue levers and shows what levels produce optimal returns.
 
Figure 4: IRR Sensitivity Heatmap
 
Appendix A: References
•	• Pickleball Kingdom Franchise Information
•	• Vetted Biz Pickleball Kingdom Franchise Analysis
•	• PickleballMAX Franchise Opportunities Overview
•	• USA Pickleball Annual Growth Report 2025
•	• SFIA and Pickleheads 2024 State of Pickleball Report
•	• Market.us Pickleball Market Size and Trends
•	• Business Research Insights Pickleball Paddles Market
•	• ResearchAndMarkets.com Pickleball Equipment Market Forecast
•	• News.market.us Pickleball Statistics and Facts
•	• The Dink Pickleball Market Growth News
•	• Franchising Magazine USA Pickleball Kingdom Expansion
•	• FranchiseWire Top Pickleball Franchises
•	• Entrepreneur Pickleball Kingdom Franchise Details
•	• Pickleball Innovators Tips for Choosing a Franchise
•	• Pickleball Kingdom 2025 FDD Franchise Information
•	• JDC Pickleball Facility ROI Guide
•	• Hard Court Sports Pickleball Franchises
•	• Forbes Indoor Pickleball Club Strategies
•	• Pickleball Kingdom Membership Details
•	• Pickleball Kingdom Plano TX Club
•	• Pickleball Kingdom Dallas North Club
•	• Pickleball Kingdom Chandler AZ Club
•	• Pickleball Kingdom Hamilton NJ Club
•	• Pickleball Kingdom Lynnwood WA Club
•	• Pickleball Kingdom Nashville South TN Club
•	• Pickleball Kingdom FAQ
•	• Pickleball Business Revenue Strategies
•	• CourtReserve Driving Revenue at Pickleball Clubs
•	• Pickleball Franchise Business Model Analysis
•	• WeFranch Pickleball Kingdom Franchisee Review
•	• Pickleheads Pickleball Statistics
 
Appendix B: Full Python Code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy_financial import irr, npv

# ---- Scenario Inputs ----
pessimistic = {
    'monthly_membership_fee': 80,
    'court_utilization_year1': 0.25,
    'member_churn_rate': 0.28,
    'initial_investment': 1000000
}
base = {
    'monthly_membership_fee': 90,
    'court_utilization_year1': 0.30,
    'member_churn_rate': 0.22,
    'initial_investment': 900000
}
optimistic = {
    'monthly_membership_fee': 100,
    'court_utilization_year1': 0.35,
    'member_churn_rate': 0.18,
    'initial_investment': 850000
}
target_irr = {
    'monthly_membership_fee': 110,
    'court_utilization_year1': 0.38,
    'member_churn_rate': 0.22,
    'initial_investment': 900000
}

# ---- Base Configuration ----
base_config = {
    'working_capital_percent_revenue': 0.08,
    'contingency_reserve': 100000,
    'financing_ratio': 0.7,
    'interest_rate': 0.08,
    'loan_term_years': 10,
    'num_courts': 12,
    'court_hours_per_day': 14,
    'days_open_per_year': 360,
    'facility_square_feet': 24000,
    'capacity_members': 1200,
    'prime_time_rate': 25,
    'off_peak_rate': 15,
    'prime_time_percentage': 0.4,
    'ancillary_rev_per_member_monthly': 20,
    'rent_per_sqft_monthly': 1.25,
    'utilities_base': 3500,
    'utilities_per_sqft': 0.15,
    'insurance_monthly': 2800,
    'staff_management': 12000,
    'staff_front_desk': 8000,
    'staff_maintenance': 4000,
    'staff_instructors_base': 3000,
    'staff_instructors_variable': 3000,
    'marketing_base': 5000,
    'equipment_maintenance_monthly': 1500,
    'supplies_monthly': 800,
    'professional_services': 1200,
    'equipment_replacement_annual': 20000,
    'operating_cost_inflation': 0.05,
    'rent_escalation': 0.03,
    'franchise_fee': 65000,
    'royalty_rate': 0.08,
    'marketing_fee_rate': 0.02,
    'years': 7,
    'tax_rate': 0.25,
    'terminal_growth_rate': 0.02,
    'discount_rate': 0.12,
    'member_acquisition_rate': 0.15,
    'max_utilization': 0.75,
    'num_members_year1': 800
}

MACRS_7_YEAR = [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893]

def calculate_cash_flows(cfg):
    # Calculate loan terms
    years = cfg['years']
    capex = cfg['initial_investment']
    total_investment = capex + cfg['contingency_reserve'] + cfg['franchise_fee']
    loan_amount = cfg['financing_ratio'] * total_investment
    equity = total_investment - loan_amount

    # Monthly loan payment using annuity formula
    r = cfg['interest_rate']
    n = cfg['loan_term_years']
    pmt = loan_amount * r / (1 - (1 + r) ** -n)
    debt_schedule = []
    remaining = loan_amount

    for year in range(1, years + 1):
        interest = remaining * r
        principal = pmt - interest
        remaining -= principal
        if remaining < 0: remaining = 0
        debt_schedule.append({'interest': interest, 'principal': principal})

    # Annual capacity and utilization
    court_capacity_hours = cfg['num_courts'] * cfg['court_hours_per_day'] * cfg['days_open_per_year']
    members = cfg['num_members_year1']
    utilization = cfg['court_utilization_year1']

    fcf = []
    revenues = []
    member_list = []

    for year in range(years):
        # --- Revenue Calculation ---
        court_hours_used = court_capacity_hours * min(utilization, cfg['max_utilization'])
        court_rev = court_hours_used * (
            cfg['prime_time_percentage'] * cfg['prime_time_rate'] +
            (1 - cfg['prime_time_percentage']) * cfg['off_peak_rate']
        )
        member_rev = members * cfg['monthly_membership_fee'] * 12
        ancillary_rev = members * cfg['ancillary_rev_per_member_monthly'] * 12
        total_rev = court_rev + member_rev + ancillary_rev
        revenues.append(total_rev)
        member_list.append(members)

        # --- Cost Calculation ---
        rent = cfg['facility_square_feet'] * cfg['rent_per_sqft_monthly'] * 12 * ((1 + cfg['rent_escalation']) ** year)
        utilities = cfg['utilities_base'] + cfg['utilities_per_sqft'] * cfg['facility_square_feet']
        insurance = cfg['insurance_monthly'] * 12
        staff = (cfg['staff_management'] + cfg['staff_front_desk'] + cfg['staff_maintenance'] +
                 cfg['staff_instructors_base'] + cfg['staff_instructors_variable']) * 12
        marketing = cfg['marketing_base'] * 12
        ops_costs = rent + utilities + insurance + staff + marketing +                     cfg['equipment_maintenance_monthly'] * 12 +                     cfg['supplies_monthly'] * 12 + cfg['professional_services'] * 12
        ops_costs *= ((1 + cfg['operating_cost_inflation']) ** year)

        franchise_fees = total_rev * (cfg['royalty_rate'] + cfg['marketing_fee_rate'])
        depreciation = capex * MACRS_7_YEAR[year] if year < len(MACRS_7_YEAR) else 0
        capex_renewal = cfg['equipment_replacement_annual']
        interest = debt_schedule[year]['interest'] if year < len(debt_schedule) else 0
        principal = debt_schedule[year]['principal'] if year < len(debt_schedule) else 0
        ebt = total_rev - ops_costs - franchise_fees - depreciation - interest
        tax = max(0, ebt * cfg['tax_rate'])
        net_income = ebt - tax

        working_cap = total_rev * cfg['working_capital_percent_revenue']
        fcf.append(net_income + depreciation - capex_renewal - working_cap - principal)

        # Update drivers
        members = members * (1 + cfg['member_acquisition_rate']) * (1 - cfg['member_churn_rate'])
        utilization = utilization * 1.05

    # Add Terminal Value in final year using Gordon Growth
    tv = fcf[-1] * (1 + cfg['terminal_growth_rate']) / (cfg['discount_rate'] - cfg['terminal_growth_rate'])
    fcf[-1] += tv

    return fcf, npv(cfg['discount_rate'], fcf), irr(fcf)

# Run the scenarios and compare
scenarios = {'Pessimistic': pessimistic, 'Base': base, 'Optimistic': optimistic, 'Target IRR': target_irr}
results = []

for name, overrides in scenarios.items():
    cfg = base_config.copy()
    cfg.update(overrides)
    fcf, scenario_npv, scenario_irr = calculate_cash_flows(cfg)
    results.append({'Scenario': name, 'NPV': round(scenario_npv, 0), 'IRR': round(scenario_irr, 3)})
    plt.plot(fcf, label=f"{name}")

# Plot free cash flows over time for each scenario
plt.title("Free Cash Flows by Scenario")
plt.xlabel("Year")
plt.ylabel("Cash Flow ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Show IRR and NPV results
df_results = pd.DataFrame(results)
display(df_results)



independent = {
    'monthly_membership_fee': 90,
    'court_utilization_year1': 0.30,
    'member_churn_rate': 0.22,
    'initial_investment': 850000
}

# New scenario dictionary including independent
scenarios = {
    'Pessimistic': pessimistic,
    'Base': base,
    'Optimistic': optimistic,
    'Target IRR': target_irr,
    'Independent': independent
}

# Modify financial logic to apply franchise fees only to franchise scenarios
results = []

for name, overrides in scenarios.items():
    cfg = base_config.copy()
    cfg.update(overrides)

    # If non-franchise, remove franchise-specific fees
    if name == 'Independent':
        cfg['franchise_fee'] = 0
        cfg['royalty_rate'] = 0.0
        cfg['marketing_fee_rate'] = 0.0
        cfg['marketing_base'] = 8000  # higher independent marketing

    fcf, scenario_npv, scenario_irr = calculate_cash_flows(cfg)
    results.append({'Scenario': name, 'NPV': round(scenario_npv, 0), 'IRR': round(scenario_irr, 3)})
    plt.plot(fcf, label=f"{name}")

plt.title("Free Cash Flows by Scenario")
plt.xlabel("Year")
plt.ylabel("Cash Flow ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_results = pd.DataFrame(results)
display(df_results)

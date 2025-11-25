"use client";
import React from "react";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import { useParams } from "next/navigation";
import GcpCsqlCostByTime from "@/components/dashboard/gcp/components/CloudSQL/GcpCsqlCostByTime";
import GcpCsqlCostByRegion from "@/components/dashboard/gcp/components/CloudSQL/GcpCsqlCostByRegion";

const costComponents = [
  {
    cloudPlatform: "gcp",
    queryType: "gcp_monthly_budget",
    costField: "gcp_fact_dim.max_monthly_budget",
    label: "Monthly Budget",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_monthly_budget_drift",
    costField: "gcp_fact_dim.monthly_budget_drift",
    label: "Monthly Budget Drift",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_month_to_date_cost",
    costField: "gcp_fact_dim.month_to_date_cost",
    label: "Month To Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_provider_name_cost",
    costField: "gcp_fact_dim.total_billed_cost",
    label: "Provider Name Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_quarterly_budget",
    costField: "gcp_fact_dim.quarterly_budget",
    label: "Quarterly Budget",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_quarterly_budget_drift",
    costField: "gcp_fact_dim.monthly_budget_drift",
    label: "Quarterly Budget Drift",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_csql_year_to_date_cost",
    costField: "gcp_fact_dim.csql_year_to_date_cost",
    label: "Csql Year to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_csql_quarter_to_date_cost",
    costField: "gcp_fact_dim.csql_quarter_to_date_cost",
    label: "Csql Quarter to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_csql_month_to_date_cost",
    costField: "gcp_fact_dim.csql_month_to_date_cost",
    label: "Csql Month to Date Cost",
  },
];

const Page: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        {costComponents.map((component, index) => (
          <CloudCostComponent
            key={index}
            cloudPlatform={component.cloudPlatform}
            queryType={component.queryType}
            costField={component.costField}
            label={component.label}
            projectId={projectId}
          />
        ))}
      </div>
      <div className="h-[300px]">
        <GcpCsqlCostByTime projectId={projectId} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="h-[300px]">
          <GcpCsqlCostByRegion projectId={projectId} />
        </div>
        <div className="h-[300px]"></div>
      </div>
    </div>
  );
};

export default Page;

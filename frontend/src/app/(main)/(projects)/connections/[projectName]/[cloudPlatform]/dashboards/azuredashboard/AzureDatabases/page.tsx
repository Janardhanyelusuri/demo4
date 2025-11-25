"use client";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import { useParams } from "next/navigation";
import React from "react";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";


type Props = {};

const costComponents = [
  // {
  //   cloudPlatform: "azure",
  //   queryType: "azure_cost_by_region_db_for_postgres",
  //   costField: "azure_fact_cost.total_billed_cost",
  //   label: "Cost by Region DB for Postgres",
  // },
  {
    cloudPlatform: "azure",
    queryType: "azure_db_for_postgres_month_to_date_cost",
    costField: "azure_fact_cost.db_for_postgres_month_to_date_cost",
    label: "Azure DB for Postgres Month to Date Cost",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_db_for_postgres_quarter_to_date_cost",
    costField: "azure_fact_cost.db_for_postgres_quarter_to_date_cost",
    label: "Azure DB for Postgres Quarter to Date Cost",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_db_for_postgres_year_to_date_cost",
    costField: "azure_fact_cost.db_for_postgres_year_to_date_cost",
    label: "Azure DB for Postgres Year to Date Cost",
  },
];

const Page = (props: Props) => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div>
      <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
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
        <div className="h-[400px] items-center justify-center flex">
          <LineChartComponent
            title="DB FOR POSTGRES COST OVER TIME"
            cloudPlatform="azure"
            queryType="azure_cost_by_time_db_for_postgres"
            projectId={projectId}
            xKey="azure_fact_cost.charge_period_start"
            yKey="azure_fact_cost.total_billed_cost"
            name="Cost by Time DB for Postgres"
            xAxisLabelFormatter={(label: string) =>
              label.length > 6 ? `${label.slice(0, 6)}...` : label
            }
            xAxisTooltipFormatter={(label: string) => label}
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
            <BarChartComponent
              title="DB FOR POSTGRES COST BY REGIONS"
              cloudPlatform="azure"
              queryType="azure_cost_by_instance_db_for_postgres"
              projectId={projectId}
              nameField="azure_resource_dim.resource_name"
              valueField="azure_fact_cost.total_billed_cost"
            />
          </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="h-[400px] items-center justify-center flex">
            <PieChartComponent
              title="DB FOR POSTGRES COST BY INSTANCES"
              cloudPlatform="azure"
              queryType="azure_cost_by_instance_db_for_postgres"
              projectId={projectId}
              nameField="azure_resource_dim.resource_name"
              valueField="azure_fact_cost.total_billed_cost"
            />
          </div>
          <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="DB FOR POSTGRES COST BY SKU METER NAME"
            cloudPlatform="azure"
            queryType="azure_cost_by_sku_db_for_postgres"
            projectId={projectId}
            nameField="azure_fact_cost.sku_meter_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div>
        </div>
      </div>
    </div>
  );
};

export default Page;

"use client";
import React from "react";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import { useParams } from "next/navigation";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import AreaChartComponent from "@/components/dashboard/charts/AreaChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";

const costComponents = [
  // {
  //   cloudPlatform: "azure",
  //   queryType: "azure_nat_gateway_month_to_date_cost",
  //   costField: "azure_fact_cost.nat_gateway_month_to_date_cost",
  //   label: "Nat Gateway Month To Date Cost",
  // },
  // {
  //   cloudPlatform: "azure",
  //   queryType: "azure_nat_gateway_quarter_to_date_cost",
  //   costField: "azure_fact_cost.nat_gateway_quarter_to_date_cost",
  //   label: "NAT Gateway Quarter To Date Cost",
  // },
  {
    cloudPlatform: "azure",
    queryType: "azure_nat_month_to_date_cost",
    costField: "azure_fact_cost.nat_month_to_date_cost",
    label: "Nat Month To Date Cost",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_nat_quarter_to_date_cost",
    costField: "azure_fact_cost.nat_quarter_to_date_cost",
    label: "Nat Quarter To Date Cost",
  },
  {
    cloudPlatform: "azure",
    queryType: "azure_nat_year_to_date_cost",
    costField: "azure_fact_cost.nat_year_to_date_cost",
    label: "Nat Year to Date Cost",
  },
  // {
  //   cloudPlatform: "azure",
  //   queryType: "azure_cost_by_instance_nat",
  //   costField: "azure_fact_cost.total_billed_cost",
  //   label: "Azure Cost by Instance Nat",
  // },
];

const CostDashboard: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="grid grid-cols-3 gap-4">
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
          title="NAT COST OVER TIME"
          cloudPlatform="azure"
          queryType="azure_cost_by_time_nat"
          projectId={projectId}
          xKey="azure_fact_cost.charge_period_start"
          yKey="azure_fact_cost.total_billed_cost"
          name="Azure Cost"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
        />
      </div>
      <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="NAT COST BY REGIONS"
            cloudPlatform="azure"
            queryType="azure_cost_by_region_nat"
            projectId={projectId}
            nameField="azure_resource_dim.region_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
        <PieChartComponent
              title="NAT COST BY INSTANCES"
              cloudPlatform="azure"
              queryType="azure_cost_by_instance_nat"
              projectId={projectId}
              nameField="azure_resource_dim.resource_name"
              valueField="azure_fact_cost.total_billed_cost"
            />
          </div>
          <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="NAT COST BY SKU METER NAME"
            cloudPlatform="azure"
            queryType="azure_cost_by_sku_nat"
            projectId={projectId}
            nameField="azure_fact_cost.sku_meter_name"
            valueField="azure_fact_cost.total_billed_cost"
          />
        </div>
        </div>
    </div>
  );
};

export default CostDashboard;

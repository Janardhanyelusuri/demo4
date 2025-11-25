"use client";

import React from "react";
import { useParams } from "next/navigation";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";

const costComponents = [
  {
    cloudPlatform: "aws",
    queryType: "load_balancing_month_to_date_cost",
    costField: "aws_fact_cost.load_balancing_month_to_date_cost",
    label: "load balancing Month to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "load_balancing_quarter_to_date_cost",
    costField: "aws_fact_cost.load_balancing_quarter_to_date_cost",
    label: "load balancing Quarter to Date Cost",
  },
  {
    cloudPlatform: "aws",
    queryType: "load_balancing_year_to_date_cost",
    costField: "aws_fact_cost.load_balancing_year_to_date_cost",
    label: "load balancing Year to Date Cost",
  },
];

const Page: React.FC = () => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
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
          title="AWS Cost by Time Load Balancer"
          cloudPlatform="aws"
          queryType="aws_cost_by_time_load_balancing"
          projectId={projectId}
          xKey="aws_fact_focus.charge_period_start"
          yKey="aws_fact_focus.total_list_cost"
          name="Cost ($)"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[300px] items-center justify-center flex">
          <PieChartComponent
            title="Aws Cost by Region Load Balancer"
            cloudPlatform="aws"
            queryType="aws_cost_by_region_load_balancing"
            projectId={projectId}
            nameField="aws_fact_focus.region_name"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
        <div className="h-[300px] items-center justify-center flex">
          <PieChartComponent
            title="Aws Cost by Instance Load Balancer"
            cloudPlatform="aws"
            queryType="aws_cost_by_instance_load_balancing"
            projectId={projectId}
            nameField="aws_fact_focus.resource_id"
            valueField="aws_fact_focus.total_list_cost"
          />
        </div>
      </div>
    </div>
  );
};

export default Page;

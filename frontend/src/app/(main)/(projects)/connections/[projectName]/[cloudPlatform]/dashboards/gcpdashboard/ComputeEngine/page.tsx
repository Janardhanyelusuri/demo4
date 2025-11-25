"use client";
import React from "react";

import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import GcpCeCostByTime from "@/components/dashboard/gcp/components/ce/GcpCeCostByTime";
import GcpCeCostByRegion from "@/components/dashboard/gcp/components/ce/GcpCeCostByRegion";
import GcpCeCostByInstance from "@/components/dashboard/gcp/components/ce/GcpCeCostByInstance";
import { useParams } from "next/navigation";

const costComponents = [
  {
    cloudPlatform: "gcp",
    queryType: "gcp_ce_quarter_to_date_cost",
    costField: "gcp_fact_dim.ce_quarter_to_date_cost",
    label: "Quarter to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_ce_year_to_date_cost",
    costField: "gcp_fact_dim.ce_year_to_date_cost",
    label: "Year to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_ce_month_to_date_cost",
    costField: "gcp_fact_dim.ce_month_to_date_cost",
    label: "Month to Date Cost",
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
      <div className="h-[400px]">
        <GcpCeCostByTime projectId={projectId} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="h-[300px]">
          <GcpCeCostByRegion projectId={projectId} />
        </div>
        <div className="h-[300px]">
          <GcpCeCostByInstance projectId={projectId} />
        </div>
      </div>
    </div>
  );
};

export default Page;

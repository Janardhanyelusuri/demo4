"use client";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import { useParams } from "next/navigation";
import React from "react";

type Props = {};

const costComponents = [
  {
    cloudPlatform: "gcp",
    queryType: "gcp_cstorage_year_to_date_cost",
    costField: "gcp_fact_dim.cstorage_year_to_date_cost",
    label: "Year to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_cstorage_quarter_to_date_cost",
    costField: "gcp_fact_dim.cstorage_quarter_to_date_cost",
    label: "Quarter to Date Cost",
  },
  {
    cloudPlatform: "gcp",
    queryType: "gcp_cstorage_month_to_date_cost",
    costField: "gcp_fact_dim.cstorage_month_to_date_cost",
    label: "Month to Date Cost",
  },
];

const Page = (props: Props) => {
  const params = useParams();
  const projectId = params.projectName as string;
  return (
    <div>
      <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
          {/* <LineChartComponent
            title=""
            cloudPlatform="gcp"
            queryType=""
            projectId={projectId}
            xKey=""
            yKey=""
            name=""
            xAxisLabelFormatter={(label: string) =>
              label.length > 6 ? `${label.slice(0, 6)}...` : label
            }
            xAxisTooltipFormatter={(label: string) => label}
          /> */}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="h-[300px] items-center justify-center flex">
            {/* <PieChartComponent
              title=""
              cloudPlatform="gcp"
              queryType=""
              projectId={projectId}
              nameField=""
              valueField=""
            /> */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Page;

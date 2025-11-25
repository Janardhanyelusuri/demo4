"use client";
import CloudCostComponent from "@/components/dashboard/charts/CloudCostComponent";
import { useParams } from "next/navigation";
import PieChartComponent from "@/components/dashboard/charts/PieChartComponent";
import LineChartComponent from "@/components/dashboard/charts/LineChartComponent";
import BarChartComponent from "@/components/dashboard/charts/BarChartComponent";
import ProjectBudgetInfo from "@/components/dashboard/charts/ProjectBudgetInfo";
import TagsBudgetInfo from "@/components/dashboard/charts/TagsBudgetInfo";

const Dashboard: React.FC = () => {
  
  const params = useParams();
  const projectId = params.projectName as string;
  
  // Ensure tag_id is treated as a string (not an array)
  const tagIdString = Array.isArray(params.tag_id) ? params.tag_id[0] : params.tag_id;
  const tagId = tagIdString ? parseInt(tagIdString, 10) : undefined; // Convert tag_id to number

  return (
    <div className="bg-cp-card shadow-md rounded-md border border-cp-border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="mb-4">
        {" "}
        <TagsBudgetInfo
          cloudPlatform="azure"
          queryType="tags_azure_budgets_data"
          projectId={projectId as string}
          title="Azure Budget"
          tag_id={tagId}
        />
      </div>
      <div className="h-[400px] items-center justify-center flex">
        <LineChartComponent
          title="COST OVER TIME"
          cloudPlatform="azure"
          queryType="azure_cost_trends_over_time_tags"
          projectId={projectId}
          xKey="azure_fact_cost.charge_period_start"
          yKey="azure_fact_cost.total_billed_cost"
          name="Azure Cost"
          xAxisLabelFormatter={(label: string) =>
            label.length > 6 ? `${label.slice(0, 6)}...` : label
          }
          xAxisTooltipFormatter={(label: string) => label}
          tag_id={tagId}
        />
      </div>    
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY SERVICE CATEGORY"
            cloudPlatform="azure"
            queryType="azure_tags_cost_by_service_category"
            projectId={projectId}
            nameField="azure_resource_dim.service_category"
            valueField="azure_fact_cost.total_billed_cost"
            tag_id={tagId}
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="COST BY RESOURCES"
            cloudPlatform="azure"
            queryType="azure_resource_name_cost_tags"
            projectId={projectId}
            nameField="azure_resource_dim.resource_name"
            valueField="azure_fact_cost.total_billed_cost"
            tag_id={tagId} // Pass the tag_id if available
          />
        </div>
      </div>
      <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY REGIONS"
            cloudPlatform="azure"
            queryType="azure_tags_cost_by_region"
            projectId={projectId}
            nameField="azure_resource_dim.region_name"
            valueField="azure_fact_cost.total_billed_cost"
            tag_id={tagId}
          />
        </div>
    </div>
  );
};

export default Dashboard;

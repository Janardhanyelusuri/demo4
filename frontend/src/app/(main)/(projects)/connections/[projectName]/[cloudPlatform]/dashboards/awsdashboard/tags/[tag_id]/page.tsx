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
    <div className="bg-white shadow-md rounded-md border w-full h-full overflow-y-auto p-4 space-y-4">
      <div className="mb-4">
        {" "}
        <TagsBudgetInfo
          cloudPlatform="aws"
          queryType="tags_aws_budgets_data"
          projectId={projectId as string}
          title="AWS Budget"
          tag_id={tagId}
        />
      </div>
      <div className="h-[400px] items-center justify-center flex">
        <LineChartComponent
          title="COST OVER TIME"
          cloudPlatform="aws"
          queryType="aws_cost_trends_over_time_tags"
          projectId={projectId}
          xKey="aws_fact_focus.charge_period_start"
          yKey="aws_fact_focus.total_list_cost"
          name="AWS Cost By Time"
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
            title=" COST BY SERVICE CATEGORY"
            cloudPlatform="aws"
            queryType="aws_tags_cost_by_service_category"
            projectId={projectId}
            nameField="aws_fact_focus.service_category"
            valueField="aws_fact_focus.total_list_cost"
            tag_id={tagId}
          />
        </div>
        <div className="h-[400px] items-center justify-center flex">
          <PieChartComponent
            title="COST BY RESOURCES"
            cloudPlatform="aws"
            queryType="aws_resource_id_cost_tags"
            projectId={projectId}
            nameField="aws_fact_focus.resource_id"
            valueField="aws_fact_focus.total_list_cost"
            tag_id={tagId} // Pass the tag_id if available
          />
        </div>
      </div>
      <div className="h-[400px] items-center justify-center flex">
          <BarChartComponent
            title="COST BY REGIONS"
            cloudPlatform="aws"
            queryType="aws_tags_cost_by_region"
            projectId={projectId}
            nameField="aws_fact_focus.region_name"
            valueField="aws_fact_focus.total_list_cost"
            tag_id={tagId}
          />
        </div>
    </div>
  );
};

export default Dashboard;

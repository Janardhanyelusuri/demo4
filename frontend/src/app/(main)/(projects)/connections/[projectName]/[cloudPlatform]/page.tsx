// tesing with comlsioe logss ------
"use client";
import React, { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { Card, CardHeader, CardDescription } from "@/components/ui/card";
import {
  Bell,
  Settings,
  LayoutDashboard,
  ArrowRight,
  Database,
  Tag,
  Trash2,
} from "lucide-react";
import {
  fetchProjectDetails,
  fetchCloudPlatformDetails,
  handleProjectDelete,
} from "@/lib/api";
import { useParams, useRouter } from "next/navigation";
import Loader from "@/components/Loader";

interface PageProps {
  params: {
    projectName: string;
    cloudPlatform: string;
  };
}

interface CustomCardProps {
  icon: React.ReactNode;
  title: string;
  href: string;
  disabled?: boolean;
}

const CustomCard: React.FC<CustomCardProps> = ({
  icon,
  title,
  href,
  disabled,
}) => (
  <Link href={disabled ? "#" : href}>
    <Card
      className={`w-[700px] bg-white shadow-md rounded-lg border border-[#E0E5EF] transition duration-200 ${
        disabled
          ? "opacity-50 cursor-not-allowed"
          : "hover:shadow-lg hover:border-[#233E7D] cursor-pointer"
      }`}
    >
      <CardHeader className="flex justify-between">
        <div className="flex items-center space-x-4 justify-between w-full">
          <div className="flex gap-2 items-center">
            {icon}
            <CardDescription className="text-[#233E7D] font-semibold text-lg">{title}</CardDescription>
          </div>
          {!disabled && <ArrowRight size={16} className="text-[#233E7D]" />}
        </div>
      </CardHeader>
    </Card>
  </Link>
);

const Page: React.FC<PageProps> = ({ params }) => {
  const { projectName, cloudPlatform } = params;
  const [projectStatus, setProjectStatus] = useState<boolean | null>(null);
  const [exportStatus, setExportStatus] = useState<boolean>(false);
  const [cloudDetails, setCloudDetails] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  const [deleteExport, setDeleteExport] = useState(false);
  const [deleteS3Bucket, setDeleteS3Bucket] = useState(false);
  const hasFetchedData = useRef(false);
  const latestDataRef = useRef<any>(null); // Reference to track latest successful API response

  const getProjectStatus = async () => {
    console.log("Fetching project status and cloud platform details...");
    setIsLoading(true);
  
    let success = false;
    let attempt = 0;
    while (!success) {
      attempt++;
      console.log(`API Call Attempt: ${attempt}`);
      try {
        // First API call
        const projectDetails = await fetchProjectDetails(projectName);
        console.log(`API Response (Project Details) on Attempt ${attempt}:`, projectDetails);
        
        if (projectDetails && projectDetails.status !== undefined) {
          console.log("Successfully fetched project details:", projectDetails);
  
          // Store the latest successful project details in a ref
          latestDataRef.current = { ...latestDataRef.current, projectStatus: projectDetails.status };
  
          // Update project status in state
          setProjectStatus(projectDetails.status);
          console.log(`State Updated: Project Status = ${projectDetails.status}`);

  
          // If project status is true, fetch cloud details
          if (projectDetails.status) {
            const cloudDetails = await fetchCloudPlatformDetails(projectName, cloudPlatform);
            
            // Store the latest successful cloud details in a ref
            latestDataRef.current = { ...latestDataRef.current, cloudDetails };
  
            // Update state for cloud details and export status
            setExportStatus(cloudDetails.export);
            setCloudDetails(cloudDetails);
            console.log(`State Updated: Cloud Details =`, cloudDetails);
            console.log(`State Updated: Export Status = ${cloudDetails.export}`);
          }
  
          // Mark as successful and exit loop
          success = true;
        } else {
          throw new Error("Invalid project details response");
        }
      } catch (error) {
        console.error("Error fetching connection status. Retrying...", error);
        console.error(`Error on Attempt ${attempt}:`, error);
        
        // Optional: Add a delay before retrying
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
  
    setIsLoading(false);
    console.log("Fetching complete. isLoading set to false.");

  };
  

  useEffect(() => {
    // Run API calls when the parameters change
    console.log("Component mounted or parameters changed, calling getProjectStatus...");
    getProjectStatus();
  }, [projectName, cloudPlatform]); // Trigger effect when params change

  const onDelete = async () => {
    try {
      await handleProjectDelete(projectName, deleteS3Bucket, deleteExport);
      router.push("/connections");
    } catch (error) {
      console.error("Error deleting connection:", error);
    }
  };

  let path = "";
  switch (cloudPlatform) {
    case "aws":
      path = "awsdashboard";
      break;
    case "gcp":
      path = "gcpdashboard";
      break;
    case "azure":
      path = "azuredashboard";
      break;
    case "snowflake":
      path = "snowflakedashboard";
      break;
    default:
      path = "";
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen w-screen">
        <Loader />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-[#F9FEFF] overflow-hidden">
      <div className="flex flex-col items-center justify-center w-full max-w-2xl mx-auto px-10 py-12">
        {projectStatus === false && exportStatus === false && (
          <div className="mb-6 p-4 border border-green-500 rounded-lg bg-green-100 text-green-700 text-center font-semibold w-full">
            Your data is currently being ingested. This may take up to 24 hours
            <br />
            Please check back later to see your dashboards.
          </div>
        )}
        {projectStatus === false && exportStatus === true && (
          <div className="mb-6 p-4 border border-yellow-500 rounded-lg bg-yellow-100 text-yellow-700 text-center font-semibold w-full">
            Data Ingestion is in progress. Come again after some time.
          </div>
        )}
        <div className="flex flex-col items-center space-y-6 mt-6 w-full">
          <CustomCard
            icon={<LayoutDashboard className="w-6 h-6 text-[#233E7D]" />}
            title="Dashboards"
            href={`/connections/${projectName}/${cloudPlatform}/dashboards/${path}/AtAGlance`}
            disabled={!projectStatus}
          />
          <CustomCard
            icon={<Bell className="w-6 h-6 text-[#233E7D]" />}
            title="Alerts Configurations"
            href={`/connections/${projectName}/${cloudPlatform}/alert_configuration/alertRules`}
            disabled={!projectStatus}
          />
          <CustomCard
            icon={<Settings className="w-6 h-6 text-[#233E7D]" />}
            title="Connection Settings"
            href={`/connections/${projectName}/${cloudPlatform}/settings`}
          />
          <CustomCard
            icon={<Database className="w-6 h-6 text-[#233E7D]" />}
            title="Manage Resources"
            href={`/connections/${projectName}/${cloudPlatform}/resources`}
          />
          <CustomCard
            icon={<Tag className="w-6 h-6 text-[#233E7D]" />}
            title="Manage Tags"
            href={`/connections/${projectName}/${cloudPlatform}/tags`}
          />
        </div>
      </div>
    </div>
  );
};

export default Page;
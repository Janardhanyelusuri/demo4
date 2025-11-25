import React, { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { fetchProjectDetails } from "@/lib/api"; // Assuming you have a function to fetch project details

// Define valid static routes
const staticRoutes = {
  landingpage: true,
  notification: true,
  onboarding: true,
  settings: true,
  login: true,
  aws: true,
  azure: true,
  gcp: true,
  snowflake: true,
  alertRules: true,
  Points: true,
  silences: true,
  AtAGlance: true,
  ec2: true,
  rds: true,
  s3: true,
  tagging: true,
};

type StaticRoute = keyof typeof staticRoutes;

export const BreadcrumbNavigation: React.FC = () => {
  const pathname = usePathname();
  const pathSegments = pathname.split("/").filter(Boolean);
  
  const [projectName, setProjectName] = useState<string | null>(null);

  // Extract projectId from the URL path
  const projectId = pathSegments[1]; // Assuming the second segment is the projectId
  const cloudPlatform = pathSegments[2]; // Assuming the third segment is the cloud platform
  
  useEffect(() => {
    if (projectId && !staticRoutes[projectId as StaticRoute]) {
      // Fetch the project name using the projectId
      const fetchProjectName = async () => {
        try {
          const projectDetails = await fetchProjectDetails(projectId);
          setProjectName(projectDetails.name); // Assuming the API response contains the project name
        } catch (error) {
          console.error("Error fetching project name:", error);
        }
      };
      fetchProjectName();
    }
  }, [projectId]);

  // Function to clean up breadcrumb segment display
  const formatSegment = (segment: string) => {
    // Replace %20 with space or replace with underscore (choose either)
    return decodeURIComponent(segment).replace(/\s/g, " "); // Replacing spaces with underscores
    // Alternatively, return decodeURIComponent(segment); for just spaces
  };

  // Generate breadcrumbs
  const breadcrumbs = pathSegments.reduce(
    (acc, segment, index) => {
      const path = `/${pathSegments.slice(0, index + 1).join("/")}`;
      
      // Always include the 'projects' segment
      if (index === 0) {
        acc.push({ segment: "Connections", path: "/connections" });
      }

      // Check if the segment is in staticRoutes or assume it's dynamic (like projectName)
if (
  !(index === 0 && segment === "connections") && // Skip duplicate Connections
  !(index === 2 && segment === cloudPlatform) // Skip cloud platform in breadcrumb
) {
  const label = index === 1 && projectName ? projectName : formatSegment(segment);
  const projectPath = index === 1 ? `${path}/${cloudPlatform}` : path;
  acc.push({ segment: label, path: projectPath });
}


      return acc;
    },
    [{ segment: "Home", path: "/landingpage" }] // Initial breadcrumb is "Home"
  );

  return (
    <Breadcrumb>
      <BreadcrumbList>
        {breadcrumbs.map((breadcrumb, index) => (
          <React.Fragment key={breadcrumb.path || index}>
            {index > 0 && (
              <BreadcrumbSeparator className="text-cp-red">
                <span className="text-[#D82026] font-bold mx-1">/</span>
              </BreadcrumbSeparator>
            )}
            <BreadcrumbItem>
              {index === breadcrumbs.length - 1 && !breadcrumb.path ? (
                <BreadcrumbPage className="font-medium text-cp-red">
                  {breadcrumb.segment}
                </BreadcrumbPage>
              ) : (
                <BreadcrumbLink
                  href={breadcrumb.path || pathname}
                  className="transition-colors text-white hover:text-cp-red"
                >
                  {breadcrumb.segment}
                </BreadcrumbLink>
              )}
            </BreadcrumbItem>
          </React.Fragment>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
};

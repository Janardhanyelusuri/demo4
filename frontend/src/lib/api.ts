import axios from "axios";
import { signOut } from "next-auth/react";

// export const BACKEND = "https://cm-api.sigmoid.io";
export const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL;

const axiosInstance = axios.create();
// Request interceptor
axiosInstance.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("accessToken");
    if (token) {
      config.headers["Authorization"] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);
// Response interceptor with auto logout on 401
axiosInstance.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      // Clear storage
      localStorage.clear();
      // Perform automatic logout
      await signOut({
        redirect: true,
        callbackUrl: "/login",
      });
    }
    return Promise.reject(error);
  }
);
export default axiosInstance;
// Define all API URLs here
const API_URLS = {
  fetchDashboards: `${BACKEND}/dashboard/`,
  fetchResourceNames: (tag_id: number) => {
    let url = `${BACKEND}/tags/tags/resources`;
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      tag_id: tag_id,
    };
  },
  deleteDashboard: (id: number) => `${BACKEND}/dashboard/${id}`,
  projects: `${BACKEND}/project/`,
  awsTestConnection: `${BACKEND}/aws/test_connection`,
  aws: `${BACKEND}/aws/`,
  alertRules: (projectId: string) => `${BACKEND}/project/${projectId}/alerts`,
  allAlertRules: () => `${BACKEND}/alert/`,
  fetchAllContactPoints: () => `${BACKEND}/integration/`,
  contactPoint: (projectId: string) =>
    `${BACKEND}/project/${projectId}/integrations`,
  tables: (projectId: string) =>
    `${BACKEND}/project/${projectId}/database-tables-columns`,
  saveAlertRule: `${BACKEND}/alert/`,
  checkProjectName: `${BACKEND}/project/check_name`,
  checkDashboardName: `${BACKEND}/dashboard/check_name`,
  testSlackConnection: `${BACKEND}/slack/test-slack-connection`,
  testTeamsConnection: `${BACKEND}/microsoft_teams/test-teams-connection`,
  // fetchData: (cloudProvider: string, queryType: string, projectId: string, granularity: string) =>
  // `${BACKEND}/queries/queries?cloud_provider=${cloudProvider}&query_type=${queryType}&project_id=${projectId}&granularity=${granularity}`,
  fetchDataOld: (cloudProvider: string) =>
    `${BACKEND}/api/v1/data?cloud_provider=${cloudProvider}`,
  gcpTestConnection: `${BACKEND}/gcp/test_connection`,
  gcp: `${BACKEND}/gcp/`,
  projectDetails: (projectId: string) => `${BACKEND}/project/${projectId}`,
  awsProjectDetails: (projectId: string) =>
    `${BACKEND}/project/${projectId}/aws`,
  azureProjectDetails: (projectId: string) =>
    `${BACKEND}/project/${projectId}/azure`,
  gcpProjectDetails: (projectId: string) =>
    `${BACKEND}/project/${projectId}/gcp`,
  snowflakeProjectDetails: (projectId: string) =>
    `${BACKEND}/project/${projectId}/snowflake`,
  deleteProject: (projectId: string) => `${BACKEND}/project/${projectId}`,
  saveIntegration: `${BACKEND}/integration/`,
  deleteAlertRule: (alertId: number) => `${BACKEND}/alert/${alertId}`,
  dashboardRequests: (projectId: string, serviceId: string) =>
    `${BACKEND}/project/${projectId}/service/${serviceId}/dashboard_requests`,
  createDashboardRequest: `${BACKEND}/dashboard_request/`,
  // fetchDashboardData: (
  // cloudProvider: string,
  // queryType: string,
  // dashboardId: string,
  // granularity: string // Added granularity parameter
  // ) =>
  // `${BACKEND}/queries/queries?cloud_provider=${cloudProvider}&query_type=${queryType}&dashboard_id=${dashboardId}&granularity=${granularity}`,
  fetchDashboardData: (
    cloudProvider: string,
    queryType: string,
    dashboardId: string,
    granularity?: string
  ) => {
    let url = `${BACKEND}/queries/queries`;
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      dashboard_id: dashboardId,
    };
    if (granularity) {
      payload.granularity = granularity; // Dynamically add granularity
    }
    // Send the POST request
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  },
  fetchData: (
    cloudProvider: string,
    queryType: string,
    projectId: string,
    granularity?: string,
    resourceResult?: string,
    service_names?: string
  ) => {
    let url = `${BACKEND}/queries/queries`;
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      project_id: projectId,
    };
    if (granularity) {
      payload.granularity = granularity; // Dynamically add granularity
    }
    if (resourceResult) {
      payload.resource_names = resourceResult; // Dynamically add resourceResult
    }
    if (service_names) {
      payload.service_names = service_names; // Dynamically add resourceResult
    }
    // Send the POST request
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  },
  fetchDataTags: (
    cloudProvider: string,
    queryType: string,
    projectId: string,
    granularity?: string,
    tag_id?: number,
    resourceResult?: string
  ) => {
    let url = `${BACKEND}/queries/queries`;
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      project_id: projectId,
    };
    if (granularity) {
      payload.granularity = granularity; // Dynamically add granularity
    }
    if (tag_id) {
      payload.tag_id = tag_id; // Dynamically add tag_id
    }
    if (resourceResult) {
      payload.resource_names = resourceResult; // Dynamically add resourceResult
    }
    // Send the POST request
    return fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  },
};
// fetchDashboardData: (
// cloudProvider: string,
// queryType: string,
// dashboardId: string,
// granularity: string // Added granularity parameter
// ) =>
// `${BACKEND}/queries/queries?cloud_provider=${cloudProvider}&query_type=${queryType}&dashboard_id=${dashboardId}&granularity=${granularity}`,
// // Add more API URLs here as needed
// };
export interface Connector {
  id: number;
  name: string;
}
export interface Dashboard {
  id: number;
  name: string;
  status: boolean;
  date: string;
  cloud_platforms: string[];
  persona: string[];
  project_ids: number[];
  connectors: {
    id: number;
    name: string;
  }[];
}
export const fetchDashboards = async (): Promise<Dashboard[]> => {
  try {
    const response = await axiosInstance.get(API_URLS.fetchDashboards, {
      headers: { Accept: "application/json" },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching dashboards:", error);
    throw new Error("Failed to fetch dashboards");
  }
};
export const deleteDashboard = async (id: number): Promise<void> => {
  try {
    await axiosInstance.delete(API_URLS.deleteDashboard(id), {
      headers: { Accept: "application/json" },
    });
  } catch (error) {
    console.error("Error deleting dashboard:", error);
    throw new Error("Failed to delete dashboard");
  }
};
export const fetchProjects = async () => {
  try {
    const response = await axiosInstance.get(API_URLS.projects);
    return response.data.sort(
      (a: { id: number }, b: { id: number }) => b.id - a.id
    );
  } catch (error) {
    console.error("Error fetching projects:", error);
    throw error;
  }
};
export const testAwsConnection = async (
  accessKey: string,
  secretKey: string
) => {
  try {
    const response = await axiosInstance.post(API_URLS.awsTestConnection, {
      aws_access_key: accessKey,
      aws_secret_key: secretKey,
    });
    return response.data;
  } catch (error) {
    console.error("Error testing AWS connection:", error);
    throw error;
  }
};
export const createProject = async (projectRequestBody: any) => {
  try {
    const response = await axiosInstance.post(
      API_URLS.projects,
      projectRequestBody,
      {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error creating project:", error);
    throw error;
  }
};
export const submitAwsFormData = async (awsRequestBody: any) => {
  try {
    const response = await axiosInstance.post(API_URLS.aws, awsRequestBody, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error submitting AWS form data:", error);
    throw error;
  }
};
export const fetchAlertRules = async (projectId: string) => {
  try {
    const response = await axiosInstance.get(API_URLS.alertRules(projectId));
    return response.data;
  } catch (error) {
    console.error("Error fetching alert rules:", error);
    throw error;
  }
};
export const fetchAllAlertRules = async () => {
  try {
    const response = await axiosInstance.get(API_URLS.allAlertRules()); // Adjust the URL function as needed
    return response.data;
  } catch (error) {
    console.error("Error fetching alert rules:", error);
    throw error;
  }
};
export const fetchAllContactPoints = async () => {
  try {
    const response = await axiosInstance.get(API_URLS.fetchAllContactPoints());
    return response.data;
  } catch (error) {
    console.error("Error fetching alert rules:", error);
    throw error;
  }
};
export const fetchContactPoints = async (projectId: string) => {
  try {
    const response = await axiosInstance.get(API_URLS.contactPoint(projectId));
    return response.data;
  } catch (error) {
    console.error("Error fetching alert rules:", error);
    throw error;
  }
};
export const fetchData = async (
  cloudProvider: string,
  queryType: string,
  projectId: string,
  granularity?: string,
  resourceResult?: string,
  service_names?: string,
  tag_id?: number,
  duration?: string // Optional duration parameter
) => {
  console.log("resourceResult", resourceResult);
  try {
    const url = `${BACKEND}/queries/queries`; // Assuming BACKEND is defined elsewhere
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      project_id: projectId,
      tag_id: tag_id,
    };
    if (granularity) {
      payload.granularity = granularity;
    }
    if (duration) {payload.duration = duration; }
    if (resourceResult) {
      payload.resource_names = resourceResult;
    }
    if (service_names) {
      payload.service_names = service_names;
    }
    console.log("Payload:", payload);
    const response = await axiosInstance.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data.data;
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
export const fetchDataTags = async (
  cloudProvider: string,
  queryType: string,
  projectId: string,
  granularity?: string,
  tag_id?: number,
  duration?: string, // Optional duration parameter
  resourceResult?: string
) => {
  console.log("resourceResult", resourceResult);
  try {
    const url = `${BACKEND}/queries/queries`; // Assuming BACKEND is defined elsewhere
    // Define payload as a dynamic object
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      project_id: projectId,
    };
    if (granularity) {
      payload.granularity = granularity;
    }
    if (tag_id !== undefined) {
      payload.tag_id = tag_id;
    }
    if (duration) {
      payload.duration = duration; // Add duration to the payload if provided
    }
    if (resourceResult) {
      payload.resource_names = resourceResult;
    }
    console.log("Payload:", payload);
    const response = await axiosInstance.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data.data;
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
export const fetchResourceNames = async (tag_id: number) => {
  try {
    const response = await axiosInstance.post(
      `${BACKEND}/tags/tags/resources`,
      { tag_id: tag_id }, // This matches the TagRequest model
      {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    // console.log(response)
    return response.data; // This should now be a comma-separated string of resource names
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
export const fetchResourceIds = async (tag_id: number) => {
  try {
    const response = await axiosInstance.post(
      `${BACKEND}/tags/tags/resource_ids`,
      { tag_id: tag_id }, // This matches the TagRequest model
      {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    // console.log(response)
    return response.data; // This should now be a comma-separated string of resource names
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
export const fetchDataOld = async (cloudProvider: string) => {
  try {
    const response = await axiosInstance.get(
      API_URLS.fetchDataOld(cloudProvider)
    );
    return response.data.data.data; // Adjust the data path as needed
  } catch (error) {
    console.error("Error fetching data:", error);
    throw error;
  }
};
export const fetchTables = async (projectId: string) => {
  try {
    const response = await axiosInstance.get(API_URLS.tables(projectId), {
      headers: {
        accept: "application/json",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching tables:", error);
    throw error;
  }
};
export const saveAlertRule = async (alertRuleData: any) => {
  try {
    const response = await axiosInstance.post(
      API_URLS.saveAlertRule,
      alertRuleData,
      {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error saving alert rule:", error);
    throw error;
  }
};
export const checkProjectName = async (projectName: string) => {
  try {
    const response = await axiosInstance.post(API_URLS.checkProjectName, {
      name: projectName,
    });
    return response.data;
  } catch (error) {
    console.error("Error checking project name:", error);
    throw error;
  }
};
export const checkDashboardName = async (projectName: string) => {
  try {
    const response = await axiosInstance.post(API_URLS.checkDashboardName, {
      name: projectName,
    });
    return response.data;
  } catch (error) {
    console.error("Error checking dashboard name:", error);
    throw error;
  }
};
export const testSlackConnection = async (webhookUrl?: string) => {
  if (!webhookUrl) return false;
  try {
    const response = await axiosInstance.post(
      API_URLS.testSlackConnection,
      {
        webhook_url: webhookUrl,
      },
      {
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
      }
    );
    return response.data.status;
  } catch (error) {
    console.error("Error testing Slack connection:", error);
    return false;
  }
};
export const testTeamsConnection = async (webhookUrl?: string) => {
  if (!webhookUrl) return false;
  try {
    const response = await axiosInstance.post(
      API_URLS.testTeamsConnection,
      {
        webhook_url: webhookUrl,
      },
      {
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
      }
    );
    return response.data.status;
  } catch (error) {
    console.error("Error testing Teams connection:", error);
    return false;
  }
};
export const saveIntegration = async (integrationData: any) => {
  try {
    const response = await axiosInstance.post(
      API_URLS.saveIntegration,
      integrationData,
      {
        headers: {
          accept: "application/json",
          "Content-Type": "application/json",
        },
      }
    );
    return response.data; // Axios automatically parses the JSON response
  } catch (error) {
    console.error("Error saving integration;", error);
    throw error;
  }
};
export const testEmailConnection = async (email?: string) => {
  if (!email) return false;
  // Implement your email connection test logic here
  // This is a placeholder implementation
  await new Promise((resolve) => setTimeout(resolve, 1000));
  return true;
};
export const deleteAlertRule = async (alertId: number) => {
  try {
    const response = await axiosInstance.delete(
      API_URLS.deleteAlertRule(alertId),
      {
        headers: {
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error deleting alert rule:", error);
    throw error;
  }
};
export const testGcpConnection = async (formData: FormData) => {
  try {
    const response = await axiosInstance.post(
      API_URLS.gcpTestConnection,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error testing GCP connection:", error);
    throw error;
  }
};
export const submitGcpFormData = async (gcpRequestBody: any) => {
  try {
    const formData = new FormData();
    for (const key in gcpRequestBody) {
      if (key === "credential_file") {
        formData.append("file", gcpRequestBody[key]);
      } else {
        formData.append(key, gcpRequestBody[key]);
      }
    }
    const response = await axiosInstance.post(API_URLS.gcp, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error submitting GCP form data:", error);
    throw error;
  }
};
export const fetchProjectDetails = async (projectId: string) => {
  try {
    const response = await axiosInstance.get(
      API_URLS.projectDetails(projectId)
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching project details:", error);
    throw error;
  }
};
// Updated fetchProjectDetails function (in your `` or relevant file)
export const newFetchProjectDetails = async (projectId: string) => {
  try {
    const response = await axiosInstance.get(
      API_URLS.projectDetails(projectId)
    );
    const data = response.data;
    return {
      name: data.name, // Project name
      cloudPlatform: data.cloud_platform, // Cloud platform associated with the project
    };
  } catch (error) {
    console.error("Error fetching project details:", error);
    return { name: "Unknown", cloudPlatform: "Unknown" }; // Default values in case of error
  }
};
export const fetchCloudPlatformDetails = async (
  projectId: string,
  cloudPlatform: string
) => {
  try {
    let url;
    switch (cloudPlatform) {
      case "aws":
        url = API_URLS.awsProjectDetails(projectId);
        break;
      case "azure":
        url = API_URLS.azureProjectDetails(projectId);
        break;
      case "gcp":
        url = API_URLS.gcpProjectDetails(projectId);
        break;
      case "snowflake":
        url = API_URLS.snowflakeProjectDetails(projectId);
        break;
      default:
        throw new Error("Invalid cloud platform");
    }
    const response = await axiosInstance.get(url);
    return response.data;
  } catch (error) {
    console.error(`Error fetching ${cloudPlatform} project details:`, error);
    throw error;
  }
};
export const handleProjectDelete = async (
  projectId: string,
  deleteS3Bucket: boolean,
  deleteExport: boolean
) => {
  try {
    const response = await axiosInstance.delete(
      API_URLS.deleteProject(projectId),
      {
        data: {
          delete_s3: deleteS3Bucket,
          delete_export: deleteExport,
        },
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error deleting project:", error);
    throw error;
  }
};
export const fetchDashboardRequests = async (
  projectId: string,
  serviceId: string
) => {
  try {
    const response = await axiosInstance.get(
      API_URLS.dashboardRequests(projectId, serviceId),
      {
        headers: {
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching dashboard requests:", error);
    throw error;
  }
};
export const createDashboardRequest = async (requestData: {
  requested_by: string;
  status: boolean;
  project_id: number;
  service_id: number;
}) => {
  try {
    const response = await axiosInstance.post(
      API_URLS.createDashboardRequest,
      requestData,
      {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      }
    );
    return response.data;
  } catch (error) {
    console.error("Error creating dashboard request:", error);
    throw error;
  }
};
export const fetchDashboardData = async (
  queryType: string,
  dashboardId: string,
  granularity?: string,
  tag_id?: number, // Add tag_id to the parameters
  duration?: string // Add duration parameter
) => {
  try {
    const url = `${BACKEND}/queries/queries`;
    // Initialize the payload
    let payload: Record<string, any> = {
      query_type: queryType,
      dashboard_id: dashboardId,
      tag_id: tag_id, // Add tag_id to the payload
    };
    // 1. Add granularity to the payload if provided
    if (granularity) {
      payload.granularity = granularity;
    }
    // 2. Add duration to the payload if provided
    if (duration) {
      payload.duration = duration;
    }
    // 3. If tag_id is provided, fetch the resource_ids
    let resourceNames = "";
    if (tag_id) {
      const resourceIds = await fetchResourceIds(tag_id); // This already returns a comma-separated string
      resourceNames = resourceIds; // Use the string directly without joining
      payload.resource_names = resourceNames; // Add resource_names to the payload
    }
    // 4. Make the request
    const response = await axiosInstance.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data.data;
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    throw error;
  }
};
export const fetchdDataDashboard = async (
  queryType: string,
  dashboardId: string,
  tag_id?: number, // Make tag_id optional
  granularity?: string
) => {
  try {
    // 1. Fetch resource names if tag_id is provided
    let resourceNames = "";
    if (tag_id) {
      const resourceIds = await fetchResourceIds(tag_id); // This already returns a comma-separated string
      resourceNames = resourceIds; // Use the string directly without joining
    }
    // 2. Construct the payload
    let payload: Record<string, any> = {
      query_type: queryType,
      dashboard_id: dashboardId,
      tag_id: tag_id, // Add tag_id to the payload
    };
    // 3. Add optional granularity if provided
    if (granularity) {
      payload.granularity = granularity;
    }
    // 4. Add resource_names to the payload if available
    if (resourceNames) {
      payload.resource_names = resourceNames;
    }
    // 5. Make the request
    const url = `${BACKEND}/queries/queries`;
    const response = await axiosInstance.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data.data;
  } catch (error) {
    console.error("Error fetching dashboard data:", error);
    throw error;
  }
};
export const ProjectfetchData = async (
  cloudProvider: string,
  queryType: string,
  projectId: string,
  granularity?: string
) => {
  try {
    const url = `${BACKEND}/queries/queries`; // Assuming BACKEND is defined elsewhere
    let payload: Record<string, any> = {
      cloud_provider: cloudProvider,
      query_type: queryType,
      project_id: projectId,
    };
    if (granularity) {
      payload.granularity = granularity;
    }
    const response = await axiosInstance.post(url, payload, {
      headers: {
        "Content-Type": "application/json",
        accept: "application/json",
      },
    });
    return response.data.data;
  } catch (error) {
    console.error("Error fetching project data:", error);
    throw error;
  }
};

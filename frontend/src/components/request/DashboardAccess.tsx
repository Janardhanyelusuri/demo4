import React, { useState, useEffect } from "react";
import { fetchDashboardRequests, createDashboardRequest } from "@/lib/api"; 
interface DashboardAccessProps {
  serviceName: string;
  serviceId: string;
  projectId: string;
}

interface DashboardRequest {
  id: number;
  requested_on: string;
  requested_by: string;
  status: boolean;
  message: string | null;
}

const DashboardAccess: React.FC<DashboardAccessProps> = ({
  serviceName,
  serviceId,
  projectId,
}) => {
  const [isRequesting, setIsRequesting] = useState(false);
  const [requestStatus, setRequestStatus] = useState<string | null>(null);
  const [existingRequest, setExistingRequest] =
    useState<DashboardRequest | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkExistingRequest = async () => {
      try {
        const requests = await fetchDashboardRequests(projectId, serviceId);
        if (requests.length > 0) {
          setExistingRequest(requests[0]);
        }
      } catch (error) {
        console.error("Error checking existing dashboard request:", error);
      } finally {
        setIsLoading(false);
      }
    };

    checkExistingRequest();
  }, [projectId, serviceId]);

  const handleRequestDashboard = async () => {
    setIsRequesting(true);
    try {
      await createDashboardRequest({
        requested_by: "User",
        status: false,
        project_id: parseInt(projectId),
        service_id: parseInt(serviceId),
      });

      setRequestStatus("Request submitted successfully.");
      // Refresh the existing request data
      const newRequests = await fetchDashboardRequests(projectId, serviceId);
      if (newRequests.length > 0) {
        setExistingRequest(newRequests[0]);
      }
    } catch (error) {
      console.error("Error requesting dashboard:", error);
      setRequestStatus("Failed to submit request. Please try again.");
    } finally {
      setIsRequesting(false);
    }
  };

  if (isLoading) {
    return <div>Loading...</div>;
  }

  return (
    <div className="bg-white shadow-md rounded-md border-[1px] w-full h-full overflow-y-auto p-4">
      <div className="flex flex-col items-center justify-center min-h-screen">
        <div className="max-w-md w-full rounded-lg p-8 space-y-6">
          <h2 className="text-3xl font-bold text-center text-gray-800">
            Dashboard: {serviceName}
          </h2>
          {existingRequest ? (
            <div className="text-center">
              <p className="text-gray-600">
                Dashboard has already been requested.
              </p>
              <p className="text-sm mt-2">
                Requested by: {existingRequest.requested_by}
              </p>
              <p className="text-sm">
                Requested on:{" "}
                {new Date(existingRequest.requested_on).toLocaleString()}
              </p>
              <p className="text-sm">
                Status: {existingRequest.status ? "Approved" : "Pending"}
              </p>
            </div>
          ) : (
            <>
              <p className="text-center text-gray-600">
                To view this dashboard, you need to  the Cloud Pulse team
              </p>
              <button
                className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition duration-300 ease-in-out transform focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                onClick={handleRequestDashboard}
                disabled={isRequesting}
              >
                {isRequesting ? "Requesting..." : "Request Dashboard"}
              </button>
              {requestStatus && (
                <p className="text-center text-sm mt-2 text-gray-600">
                  {requestStatus}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardAccess;

// import React, { useEffect, useState } from "react";
// import { Dialog, Transition } from "@headlessui/react";
// import { Button } from "@/components/ui/button";
// import { useToast } from "@/components/ui/use-toast";
// import { Loader } from "lucide-react";
// import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
// import { ChevronDownIcon } from "@heroicons/react/20/solid";
// import axiosInstance, { fetchProjects, checkProjectName } from "@/lib/api";
// import NewAlertRuleForm from "./NewAlertRuleForm";


// interface ConnectionAlertRuleProps {
//  onBack: () => void;
//  onSave: () => void;
//  projectId: string;
//  selectedTagIds?: number[]; 
// }


// interface CloudProvider {
//  id: number;
//  name: string;
// }


// interface Connection {
//  id: number;
//  name: string;
//  cloud_platform: "aws" | "gcp" | "azure" | "snowflake";
//  date: string;
// }


// interface Project {
//  id: string;
//  name: string;
//  cloud_platform: "aws" | "gcp" | "azure" | "snowflake";
//  date: string;
// }


// const ConnectionAlertRule: React.FC<ConnectionAlertRuleProps> = ({ onBack, onSave, projectId ,selectedTagIds}) => {
//  const { toast } = useToast();
//  const [projects, setProjects] = useState<Connection[]>([]); // Store connections as projects
//  const [alertName, setAlertName] = useState("");
//  const [cloudProviders, setCloudProviders] = useState<CloudProvider[]>([
//    { id: 1, name: "AWS" },
//    { id: 2, name: "GCP" },
//    { id: 3, name: "Azure" },
//  ]); // Hardcoded cloud providers
//  const [selectedCloudProvider, setSelectedCloudProvider] = useState<CloudProvider | null>(null);
//  const [filteredConnections, setFilteredConnections] = useState<Connection[]>([]); // Store filtered connections
//  const [selectedConnection, setSelectedConnection] = useState<Connection | null>(null);
//  const [isDialogOpen, setIsDialogOpen] = useState(true);
//  const [loading, setLoading] = useState(false);


//  // Fetch all connections (projects)
//  useEffect(() => {
//    const getProjects = async () => {
//      try {
//        const fetchedProjects = await fetchProjects();
//        setProjects(fetchedProjects);
//      } catch (error) {
//        console.error("Error fetching connections:", error);
//      }
//    };


//    getProjects();
//  }, []);


//  // Handle cloud selection
//  const handleCloudSelect = (cloud: CloudProvider) => {
//    setSelectedCloudProvider(cloud);
//    setSelectedConnection(null); // Reset selected connection


//    // Filter connections by selected cloud provider
//    const filtered = projects.filter((connection) => connection.cloud_platform === cloud.name.toLowerCase());
//    setFilteredConnections(filtered);
//  };


//  // Handle connection selection
//  const handleConnectionSelect = (connection: Connection) => {
//    setSelectedConnection(connection);
//  };


//  // Proceed to form after connection selection
//  const handleProceedToForm = () => {
//    if (!selectedConnection) {
//      toast({
//        title: "Error",
//        description: "Please select a connection",
//        variant: "destructive",
//      });
//      return;
//    }
//    setIsDialogOpen(false); // Close the dialog and proceed to form
//  };


//  return (
//    <>
//      {/* Dialog for Cloud and Connection Selection */}
//      <Transition show={isDialogOpen}>
//      <Dialog onClose={onBack} className="relative z-50">
//      <Transition.Child>
//            <div className="fixed inset-0 bg-black bg-opacity-25"></div>
//          </Transition.Child>
//          <div className="fixed inset-0 flex items-center justify-center p-4">
//            <Transition.Child>
//              <Dialog.Panel className="w-full max-w-md p-6 bg-white rounded">
//                <Dialog.Title className="text-xl font-semibold">Select Cloud Provider</Dialog.Title>


//                {/* Cloud Dropdown */}
//                <Menu as="div" className="relative inline-block text-left mt-4">
//                  <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
//                    {selectedCloudProvider ? selectedCloudProvider.name : "Select Cloud Provider"}
//                    <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
//                  </MenuButton>
//                  <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
//                    {cloudProviders.map((cloud) => (
//                      <MenuItem key={cloud.id}>
//                        {({ active }) => (
//                          <button
//                            className={`${
//                              active ? "bg-blue-100" : ""
//                            } block w-full text-left px-4 py-2 text-sm`}
//                            onClick={() => handleCloudSelect(cloud)}
//                          >
//                            {cloud.name}
//                          </button>
//                        )}
//                      </MenuItem>
//                    ))}
//                  </MenuItems>
//                </Menu>


//                {/* Connection Dropdown (Shown after Cloud is selected) */}
//                {selectedCloudProvider && (
//                  <>
//                    <Dialog.Title className="text-xl font-semibold mt-4">Select Connection</Dialog.Title>
//                    <Menu as="div" className="relative inline-block text-left mt-4">
//                      <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
//                        {selectedConnection ? selectedConnection.name : "Select Connection"}
//                        <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
//                      </MenuButton>
//                      <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none max-w-full">
//                       {filteredConnections.length === 0 ? (
//                         <div className="px-4 py-2 text-sm text-gray-500">No connections available for selected cloud</div>
//                       ) : (
//                         filteredConnections.map((connection) => (
//                           <MenuItem key={connection.id}>
//                             {({ active }) => (
//                               <button
//                                 className={`${
//                                   active ? "bg-blue-100" : ""
//                                 } block w-full text-left px-4 py-2 text-sm overflow-hidden text-ellipsis whitespace-nowrap`}
//                                 style={{ maxWidth: '100%' }}  // Ensures the width is responsive and prevents overflow
//                                 onClick={() => handleConnectionSelect(connection)}
//                               >
//                                 {connection.name}
//                               </button>
//                             )}
//                           </MenuItem>
//                         ))
//                       )}
//                     </MenuItems>

//                    </Menu>
//                  </>
//                )}


//                {/* Proceed Button */}
//                <div className="mt-4">
//                  <Button variant="default" onClick={handleProceedToForm}>
//                    Proceed to Form
//                  </Button>
//                </div>
//              </Dialog.Panel>
//            </Transition.Child>
//          </div>
//        </Dialog>
//      </Transition>


//      {/* The Rest of the Form (shown after dialog is closed) */}
//      {!isDialogOpen && selectedConnection && (  // Check if selectedConnection exists before rendering
//        <div className="mt-4">
//         {selectedTagIds && selectedTagIds.length > 0 && (
//   <div className="mb-4">
//     {/* <h3 className="text-lg font-semibold">Selected Tags:</h3>
//     <ul className="list-disc pl-5">
//       {selectedTagIds.map((tagId) => (
//         <li key={tagId} className="text-gray-700">
//           Tag ID: {tagId}
//         </li>
//       ))}
//     </ul> */}
//   </div>
// )}
//          <NewAlertRuleForm
//            projectId={selectedConnection.id.toString()} // Pass the selected connection ID
//            onBack={onBack}
//            onSave={onSave}
//            selectedTagIds={selectedTagIds}
//          />
//        </div>
//      )}
//    </>
//  );
// };


// export default ConnectionAlertRule;

import React, { useEffect, useState } from "react";
import { Dialog, Transition } from "@headlessui/react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/use-toast";
import { Loader, X } from "lucide-react"; // Import X icon
import { Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { ChevronDownIcon } from "@heroicons/react/20/solid";
import axiosInstance, { fetchProjects, checkProjectName } from "@/lib/api";
import NewAlertRuleForm from "./NewAlertRuleForm";

interface ConnectionAlertRuleProps {
  onBack: () => void;
  onSave: () => void;
  projectId: string;
  selectedTagIds?: number[]; 
}

interface CloudProvider {
  id: number;
  name: string;
}

interface Connection {
  id: number;
  name: string;
  cloud_platform: "aws" | "gcp" | "azure" | "snowflake";
  date: string;
}

interface Project {
  id: string;
  name: string;
  cloud_platform: "aws" | "gcp" | "azure" | "snowflake";
  date: string;
}

const ConnectionAlertRule: React.FC<ConnectionAlertRuleProps> = ({ 
  onBack, 
  onSave, 
  projectId,
  selectedTagIds
}) => {
  const { toast } = useToast();
  const [projects, setProjects] = useState<Connection[]>([]);
  const [cloudProviders] = useState<CloudProvider[]>([
    { id: 1, name: "AWS" },
    { id: 2, name: "GCP" },
    { id: 3, name: "Azure" },
  ]);
  const [selectedCloudProvider, setSelectedCloudProvider] = useState<CloudProvider | null>(null);
  const [filteredConnections, setFilteredConnections] = useState<Connection[]>([]);
  const [selectedConnection, setSelectedConnection] = useState<Connection | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(true);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const getProjects = async () => {
      try {
        const fetchedProjects = await fetchProjects();
        setProjects(fetchedProjects);
      } catch (error) {
        console.error("Error fetching connections:", error);
      }
    };

    getProjects();
  }, []);

  const handleCloudSelect = (cloud: CloudProvider) => {
    setSelectedCloudProvider(cloud);
    setSelectedConnection(null);
    const filtered = projects.filter(
      connection => connection.cloud_platform === cloud.name.toLowerCase()
    );
    setFilteredConnections(filtered);
  };

  const handleConnectionSelect = (connection: Connection) => {
    setSelectedConnection(connection);
  };

  const handleProceedToForm = () => {
    if (!selectedConnection) {
      toast({
        title: "Error",
        description: "Please select a connection",
        variant: "destructive",
      });
      return;
    }
    setIsDialogOpen(false);
  };

  // ADDED: Function to close dialog and trigger back action
  const handleCloseDialog = () => {
    setIsDialogOpen(false);
    onBack();
  };

  return (
    <>
      <Transition show={isDialogOpen}>
        <Dialog onClose={onBack} className="relative z-50">
          <Transition.Child>
            <div className="fixed inset-0 bg-black bg-opacity-25"></div>
          </Transition.Child>
          <div className="fixed inset-0 flex items-center justify-center p-4">
            <Transition.Child>
              <Dialog.Panel className="w-full max-w-md p-6 bg-white rounded-md relative border border-[#E0E5EF]">
                <button
                  onClick={handleCloseDialog}
                  className="absolute top-4 right-4 p-1 rounded-full hover:bg-[#F9FEFF]"
                >
                  <X className="h-5 w-5 text-[#6B7280]" />
                </button>
                <Dialog.Title className="text-xl font-semibold text-[#233E7D]">
                  Select Cloud Provider
                </Dialog.Title>

                {/* Cloud Dropdown */}
                <Menu as="div" className="relative inline-block text-left mt-4">
                  <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-[#233E7D] shadow-sm ring-1 ring-inset ring-[#C8C8C8] hover:bg-[#233E7D]/5">
                    {selectedCloudProvider ? selectedCloudProvider.name : "Select Cloud Provider"}
                    <ChevronDownIcon className="-mr-1 h-5 w-5 text-[#6B7280]" aria-hidden="true" />
                  </MenuButton>
                  <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-[#E0E5EF] ring-opacity-100 focus:outline-none">
                    {cloudProviders.map((cloud) => (
                      <MenuItem key={cloud.id}>
                        {({ active }) => (
                          <button
                            className={`${
                              active ? "bg-[#233E7D]/10" : ""
                            } block w-full text-left px-4 py-2 text-sm text-[#233E7D]`}
                            onClick={() => handleCloudSelect(cloud)}
                          >
                            {cloud.name}
                          </button>
                        )}
                      </MenuItem>
                    ))}
                  </MenuItems>
                </Menu>

                {/* Connection Dropdown */}
                {selectedCloudProvider && (
                  <>
                    <Dialog.Title className="text-xl font-semibold mt-4">
                      Select Connection
                    </Dialog.Title>
                    <Menu as="div" className="relative inline-block text-left mt-4">
                      <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
                        {selectedConnection ? selectedConnection.name : "Select Connection"}
                        <ChevronDownIcon className="-mr-1 h-5 w-5 text-gray-400" aria-hidden="true" />
                      </MenuButton>
                      <MenuItems className="absolute z-10 mt-2 w-full origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none max-w-full">
                        {filteredConnections.length === 0 ? (
                          <div className="px-4 py-2 text-sm text-gray-500">
                            No connections available for selected cloud
                          </div>
                        ) : (
                          filteredConnections.map((connection) => (
                            <MenuItem key={connection.id}>
                              {({ active }) => (
                                <button
                                  className={`${
                                    active ? "bg-blue-100" : ""
                                  } block w-full text-left px-4 py-2 text-sm overflow-hidden text-ellipsis whitespace-nowrap`}
                                  style={{ maxWidth: '100%' }}
                                  onClick={() => handleConnectionSelect(connection)}
                                >
                                  {connection.name}
                                </button>
                              )}
                            </MenuItem>
                          ))
                        )}
                      </MenuItems>
                    </Menu>
                  </>
                )}

                {/* Proceed Button */}
                <div className="mt-4">
                  <Button variant="default" onClick={handleProceedToForm}>
                    Proceed to Form
                  </Button>
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </Dialog>
      </Transition>

      {/* Form section */}
      {!isDialogOpen && selectedConnection && (
        <div className="mt-4">
          <NewAlertRuleForm
            projectId={selectedConnection.id.toString()}
            onBack={onBack}
            onSave={onSave}
            selectedTagIds={selectedTagIds}
          />
        </div>
      )}
    </>
  );
};

export default ConnectionAlertRule;
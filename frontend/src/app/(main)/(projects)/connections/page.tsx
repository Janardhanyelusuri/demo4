"use client";
import React, { useEffect, useState } from "react";
import AWS from "@/assets/AWS.svg";
import GOOGLE from "@/assets/google.svg";
import AZURE from "@/assets/azure.svg";
import SNOWFLAKE from "@/assets/snowflake.svg";
import Image from "next/image";
import Link from "next/link";
import { RefreshCcw, CirclePlus } from "lucide-react";
import { EllipsisVertical, Trash, Plus } from "lucide-react";
import axiosInstance, { fetchProjects, checkProjectName, BACKEND } from "@/lib/api";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";
import { useRouter, useParams } from "next/navigation";
import ConnectionAlertRule from "@/components/alertRules/ConnectionAlertRule";

// Define the Project type
interface Project {
  id: string;
  name: string;
  cloud_platform: "aws" | "gcp" | "azure";
  date: string;
}

type CloudPlatform = 'aws' | 'gcp' | 'azure';

const Page = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [deleteAlert, setDeleteAlert] = useState({ show: false, message: "" });
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [syncDialogOpen, setSyncDialogOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null);
  const [projectToDeletePlatform, setProjectToDeletePlatform] = useState<string | null>(null);
  const [projectToSync, setProjectToSync] = useState<string | null>(null);
  const [deleteExport, setDeleteExport] = useState(false);
  const [deleteS3Bucket, setDeleteS3Bucket] = useState(false);
  const [deleteContainer, setDeleteContainer] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [inputError, setInputError] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isNameChecked, setIsNameChecked] = useState(false);
  const [isNameValid, setIsNameValid] = useState(false);
  const [showNewAlertForm, setShowNewAlertForm] = useState(false);
  const { toast } = useToast();
  const router = useRouter();
  const params = useParams();
  const projectId = params.projectName as string;
  const [isButtonVisible, setIsButtonVisible] = useState(true);
  const [selectedPlatform, setSelectedPlatform] = useState<CloudPlatform | null>(null);
  const [alertRuleData, setAlertRuleData] = useState<any>(null);

  const hasFetched = React.useRef(false);
  useEffect(() => {
    if (hasFetched.current) return;
    hasFetched.current = true;
    const getProjects = async () => {
      try {
        const sortedProjects = await fetchProjects();
        setProjects(sortedProjects);
      } catch (error) {
        console.error("Error fetching :", error);
      }
    };
    getProjects();
  }, []);

  const cloudImages: Record<CloudPlatform, any> = {
    aws: AWS,
    gcp: GOOGLE,
    azure: AZURE,
    // snowflake: SNOWFLAKE,
  };

  const groupProjectsByPlatform = () => {
    const grouped: Record<CloudPlatform, Project[]> = {
      aws: [],
      gcp: [],
      azure: [],
      // snowflake: []
    };
  
    projects.forEach(project => {
      grouped[project.cloud_platform].push(project);
    });
  
    return grouped;
  };
  
  const handleDeleteClick = (
    id: string,
    platform: string,
    event: React.MouseEvent
  ) => {
    event.preventDefault();
    event.stopPropagation();
    setProjectToDelete(id);
    setProjectToDeletePlatform(platform);
    setDeleteDialogOpen(true);
  };

  const handleDelete = async () => {
    if (!projectToDelete) return;
  
    let deleteData: any = {};
  
    if (projectToDeletePlatform === "aws") {
      deleteData = {
        delete_export: deleteExport,
        delete_s3: deleteS3Bucket,
      };
    } else if (projectToDeletePlatform === "azure") {
      deleteData = {
        delete_export: deleteExport,
        delete_container: deleteContainer,
      };
    }
  
    try {
      const response = await axiosInstance.delete(`${BACKEND}/project/${projectToDelete}`, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
        data: deleteData,
      });
  
      if (response.status === 200) {
        setProjects(
          projects.filter((project) => project.id !== projectToDelete)
        );
      } else {
        throw new Error("Failed to delete connection");
      }
    } catch (error) {
      console.error("Error deleting connection:", error);
      setDeleteAlert({ show: true, message: "Failed to delete connection" });
    }
  
    setDeleteDialogOpen(false);
    setProjectToDelete(null);
    setProjectToDeletePlatform(null);
    setDeleteExport(false);
    setDeleteS3Bucket(false);
    setDeleteContainer(false);
  
    setTimeout(() => setDeleteAlert({ show: false, message: "" }), 3000);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.replace(/\s/g, "");
    setProjectName(newValue);
    setIsNameChecked(false);
  };

// Update handleCheckNameAndContinue function
const handleCheckNameAndContinue = async () => {
  if (projectName.trim() === "") {
    setInputError(true);
    return;
  }

  try {
    const data = await checkProjectName(projectName.trim());

    if (data.status) {
      setIsNameValid(true);
      setInputError(false);
      // Standardize platform case before sending
      const platformParam = selectedPlatform ? 
        selectedPlatform === 'azure' ? 'Azure' :
        selectedPlatform.toUpperCase() : '';
      
      const url = `/onboarding?projectName=${encodeURIComponent(projectName.trim())}${
        platformParam ? `&platform=${platformParam}` : ''
      }`;
      router.push(url);
    } else {
      setIsNameValid(false);
      toast({
        title: "Error",
        description: "Connection with the same name already exists. Please try another name.",
      });
      setInputError(true);
    }

    setIsNameChecked(true);
  } catch (error) {
    console.error("Error checking connection name:", error);
  }
};

  const toggleDialog = () => {
    setIsDialogOpen(!isDialogOpen);
    if (!isDialogOpen) {
      setInputError(false);
      setIsNameChecked(false);
      setIsNameValid(false);
    }
  };

  const handleSync = async () => {
    if (!projectToSync) return;
  
    try {
      const response = await axiosInstance.get(`${BACKEND}/project/${projectToSync}/run_ingestion`, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });
  
      if (response.status === 200) {
        toast({
          title: "Success",
          description: "Connection synced successfully.",
        });
      } else {
        throw new Error("Failed to sync connection");
      }
    } catch (error) {
      console.error("Error syncing connection:", error);
      toast({
        title: "Error",
        description: "Failed to sync connection.",
      });
    }
  
    setSyncDialogOpen(false);
    setProjectToSync(null);
  };

  const handleNewAlertClick = () => {
    setShowNewAlertForm(true);
    setIsButtonVisible(false);
  };
  
  const handleBackClick = () => {
    setShowNewAlertForm(false);
    setIsButtonVisible(true);
  };
  
  const handleSaveAlertRule = () => {
    if (alertRuleData) {
      saveAlertRule(alertRuleData);
    }
  };

  const saveAlertRule = async (data: any) => {
    try {
      const response = await axiosInstance.post(`${BACKEND}/project/${projectId}/alert-rules`, data);
      
      if (response.status === 200) {
        toast({
          title: "Success",
          description: "Alert rule saved successfully.",
        });
        setShowNewAlertForm(false);
        setAlertRuleData(null);
      } else {
        throw new Error("Failed to save alert rule");
      }
    } catch (error) {
      console.error("Error saving alert rule:", error);
      toast({
        title: "Error",
        description: "Failed to save alert rule.",
      });
    }
  };

  return (
    <div className="mt-4 pl-4 h-full bg-[#F9FEFF]">
      <div className="flex justify-between items-center mb-6">
        <div className="text-2xl font-bold text-[#233E7D]">
          {selectedPlatform ? `${selectedPlatform.toUpperCase()} Connections` : 'Connections'}
        </div>
        {isButtonVisible && (
          <Button
            onClick={handleNewAlertClick}
            className="inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#D82026] focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-[#D82026] text-white hover:bg-[#b81a1f] h-10 px-4 py-2 gap-2 mr-10"
          >
            <Plus className="h-4 w-4" />
            New Alert Rule
          </Button>
        )}
      </div>

      {!showNewAlertForm ? (
        <>
          {deleteAlert.show && (
            <Alert className="mb-4 rounded-md border border-[#E0E5EF] bg-white">
              <AlertTitle className="text-[#233E7D] font-semibold">Info</AlertTitle>
              <AlertDescription className="text-base text-[#6B7280]">{deleteAlert.message}</AlertDescription>
            </Alert>
          )}

          <div className="flex flex-wrap -m-2 overflow-auto h-full">
            {!selectedPlatform ? (
              <>
                {/* Add New Connection Card */}
                <div
                  className="bg-white rounded-md shadow-md border border-transparent p-4 mt-4 flex flex-col items-center justify-center text-center m-2 cursor-pointer hover:shadow-lg transition-shadow hover:border-[#233E7D]"
                  style={{ width: "200px", height: "150px" }}
                  onClick={toggleDialog}
                >
                  <Plus className="text-[#233E7D]" size={48} />
                  <h2 className="mt-2 mb-2 text-[#233E7D] text-lg font-semibold">
                    Add Connection
                  </h2>
                </div>

                {/* Platform Cards */}
                {Object.entries(cloudImages).map(([platform, icon]: [string, any]) => (
                  <div
                    key={platform}
                    className="bg-white rounded-md shadow-md border border-transparent p-4 mt-4 flex flex-col items-center justify-center text-center m-2 cursor-pointer hover:shadow-lg transition-shadow hover:border-[#233E7D]"
                    style={{ width: "200px", height: "150px" }}
                    onClick={() => setSelectedPlatform(platform as CloudPlatform)}
                  >
                    <Image
                      src={icon}
                      alt={platform}
                      width={48}
                      height={48}
                    />
                    <h2 className="mt-2 font-semibold mb-2 text-lg text-[#233E7D]">
                      {platform.toUpperCase()}
                    </h2>
                    <span className="text-sm text-[#6B7280]">
                      {groupProjectsByPlatform()[platform as CloudPlatform].length} Projects
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <div className="w-full">
                <div className="flex items-center mb-4">
                  <Button
                    variant="ghost"
                    onClick={() => setSelectedPlatform(null)}
                    className="flex items-center gap-2 text-[#233E7D] hover:text-[#233E7D] text-sm font-medium"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="m15 18-6-6 6-6" />
                    </svg>
                    Back to Platforms
                  </Button>
                </div>
  
                <div className="flex flex-wrap gap-4">
                  <div
                    className="bg-white rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col items-center justify-center text-center hover:border-[#233E7D] cursor-pointer transition-colors"
                    style={{ width: "200px", height: "150px" }}
                    onClick={toggleDialog}
                  >
                    <Plus className="text-[#233E7D]" size={48} />
                    <h2 className="mt-2 font-semibold mb-2 text-sm text-[#233E7D]">
                      Add {selectedPlatform.toUpperCase()} Connection
                    </h2>
                  </div>
  
                  {groupProjectsByPlatform()[selectedPlatform].map(({ id, name, cloud_platform, date }) => (
                    <Link href={`/connections/${id}/${cloud_platform}`} key={id}>
                      <div
                        className="bg-white rounded-md shadow-md border border-[#E0E5EF] p-4 flex flex-col justify-between hover:border-[#233E7D]"
                        style={{ width: "200px", height: "150px" }}
                      >
                        <div>
                          <div className="flex justify-between items-center">
                            <Image
                              src={cloudImages[cloud_platform]}
                              alt={cloud_platform}
                              width={32}
                              height={32}
                            />
                            <DropdownMenu>
                              <DropdownMenuTrigger>
                                <EllipsisVertical className="cursor-pointer text-[#233E7D]" />
                              </DropdownMenuTrigger>
                              <DropdownMenuContent>
                                <DropdownMenuItem
                                  onClick={(e) => handleDeleteClick(id, cloud_platform, e)}
                                  className="text-[#D82026] hover:bg-[#D82026]/10"
                                >
                                  <Trash className="mr-2 h-4 w-4" />
                                  <span>Delete</span>
                                </DropdownMenuItem>
                                <DropdownMenuItem
                                  onClick={(e) => {
                                    e.preventDefault();
                                    e.stopPropagation();
                                    setProjectToSync(id);
                                    setSyncDialogOpen(true);
                                  }}
                                  className="text-[#233E7D] hover:bg-[#233E7D]/10"
                                >
                                  <RefreshCcw className="mr-2 h-4 w-4" />
                                  <span>Sync</span>
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                          <h2 className="mt-2 font-semibold mb-2 text-sm truncate text-[#233E7D]">
                            {name}
                          </h2>
                        </div>
                        <div className="flex flex-col text-xs text-[#6B7280]">
                          <span>Last edited on</span>
                          <div className="flex justify-between items-center w-full">
                            <span className="text-xs">{date}</span>
                            <div className="w-2 h-2 mr-2 bg-green-500 rounded-full"></div>
                          </div>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Dialogs */}
          <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
            <DialogContent className="rounded-md bg-white">
              <DialogHeader>
                <DialogTitle className="text-[#D82026] text-2xl font-bold">Delete Connection</DialogTitle>
                <DialogDescription className="text-base text-[#6B7280]">
                  Are you sure you want to delete this connection? This action cannot
                  be undone.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                {projectToDeletePlatform === "aws" && (
                  <>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="deleteExport"
                        checked={deleteExport}
                        onCheckedChange={(checked) =>
                          setDeleteExport(checked as boolean)
                        }
                      />
                      <Label htmlFor="deleteExport" className="text-[#233E7D] text-sm">Delete Export</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="deleteS3Bucket"
                        checked={deleteS3Bucket}
                        onCheckedChange={(checked) =>
                          setDeleteS3Bucket(checked as boolean)
                        }
                      />
                      <Label htmlFor="deleteS3Bucket" className="text-[#233E7D] text-sm">Delete S3 Bucket</Label>
                    </div>
                  </>
                )}
                {projectToDeletePlatform === "azure" && (
                  <>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="deleteExport"
                        checked={deleteExport}
                        onCheckedChange={(checked) =>
                          setDeleteExport(checked as boolean)
                        }
                      />
                      <Label htmlFor="deleteExport" className="text-[#233E7D] text-sm">Delete Export</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="deleteContainer"
                        checked={deleteContainer}
                        onCheckedChange={(checked) =>
                          setDeleteContainer(checked as boolean)
                        }
                      />
                      <Label htmlFor="deleteContainer" className="text-[#233E7D] text-sm">Delete Container</Label>
                    </div>
                  </>
                )}
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setDeleteDialogOpen(false)} className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md">
                  Cancel
                </Button>
                <Button variant="destructive" onClick={handleDelete} className="bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md">
                  Delete
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={syncDialogOpen} onOpenChange={setSyncDialogOpen}>
            <DialogContent className="rounded-md bg-white">
              <DialogHeader>
                <DialogTitle className="text-[#233E7D] text-2xl font-bold">Sync Connection</DialogTitle>
                <DialogDescription className="text-base text-[#6B7280]">
                  Are you sure you want to sync this connection? This action will update
                  the connection with the latest data.
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={() => setSyncDialogOpen(false)} className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md">
                  Cancel
                </Button>
                <Button variant="default" onClick={handleSync} className="bg-[#233E7D] text-white hover:bg-[#19294e] text-sm font-medium rounded-md">
                  Sync
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          <Dialog open={isDialogOpen} onOpenChange={toggleDialog}>
            <DialogContent className="sm:max-w-[550px] rounded-md bg-white">
              <DialogHeader>
                <DialogTitle className="text-[#233E7D] text-2xl font-bold">New Connection</DialogTitle>
                <DialogDescription className="text-base text-[#6B7280]">
                  Enter the name of your new connection. Click continue when you&apos;re
                  done.
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-1 items-center gap-4">
                  <Input
                    id="connection-name"
                    value={projectName}
                    onChange={handleInputChange}
                    className={`w-full border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base ${inputError ? "border-[#D82026]" : ""}`}
                    placeholder="Name of Your Connection"
                  />

                  <div className="space-y-3">
                    {/* Naming Convention Section */}
                    <div>
                      <p className="text-xs text-[#6B7280] font-semibold mb-1">Naming Convention:</p>
                      <p className="text-xs text-[#6B7280] mb-2">
                        Use underscores (_) to separate words. Spaces are not allowed.
                      </p>
                      <p className="text-xs font-mono bg-[#F9FEFF] p-2 rounded">
                        &lt;platform&gt;_&lt;connectorType&gt;_&lt;environment&gt;_&lt;region&gt;_&lt;version&gt;
                      </p>
                    </div>

                    {/* Component Definitions */}
                    <div className="text-xs text-[#6B7280]">
                      <p className="font-semibold mb-1">Components:</p>
                      <ul className="list-disc list-inside space-y-1 pl-2">
                        <li><span className="font-mono">platform</span>: aws, gcp, azure</li>
                        <li><span className="font-mono">connectorType</span>: s3, bigquery, storage</li>
                        <li><span className="font-mono">environment</span>: prod, dev, stage</li>
                        <li><span className="font-mono">region</span>: use1 (US East 1), usw2 (US West 2)</li>
                        <li><span className="font-mono">version</span>: v1, v2, etc.</li>
                      </ul>
                    </div>

                    {/* Examples */}
                    <div className="text-xs text-[#6B7280]">
                      <p className="font-semibold mb-1">Examples:</p>
                      <ul className="list-disc list-inside space-y-1 pl-2">
                        <li><span className="font-mono">aws_s3_prod_use1_v1</span></li>
                        <li><span className="font-mono">gcp_bigquery_prod_use1_v2</span></li>
                      </ul>
                    </div>
                  </div>

                  {inputError && (
                    <p className="text-[#D82026] text-xs">Connection name is required</p>
                  )}
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={toggleDialog} className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md">
                  Cancel
                </Button>
                <Button variant="default" onClick={handleCheckNameAndContinue} className="bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md">
                  Continue
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </>
      ) : (
        <ConnectionAlertRule
          onBack={handleBackClick}
          projectId={projectId}
          onSave={handleSaveAlertRule}
        />
      )}
    </div>
  );
};

export default Page;
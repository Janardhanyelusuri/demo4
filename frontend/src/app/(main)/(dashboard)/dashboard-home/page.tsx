"use client";
import React, { useEffect, useState, useMemo } from "react";
import Image from "next/image";
import {
  Megaphone,
  Share2,
  Trash2,
  ChevronDown,
  ChevronUp,
  Plus,
  RefreshCcw, 
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import AZURE from "@/assets/azure.svg";
import AWS from "@/assets/AWS.svg";
import GCP from "@/assets/google.svg";
import SNOWFLAKE from "@/assets/snowflake.svg";
import axiosInstance, { deleteDashboard, fetchDashboards, checkDashboardName, BACKEND } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";

interface Connector {
  id: number;
  name: string;
}

interface Dashboard {
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

const PERSONA_OPTIONS = [
  { value: "Executives: CEO/CTO", label: "Executives: CEO/CTO" },
  { value: "FinOps Team", label: "FinOps Team" },
  { value: "Product Owner", label: "Product Owner" },
];

// New interface for grouped dashboards
interface GroupedDashboard {
  name: string;
  dashboards: Dashboard[]; // Store all dashboards in this group
  cloud_platforms: string[];
  personas: string[]; // All personas from all dashboards in the group
  connectors: {
    id: number;
    name: string;
  }[];
  date: string;
}

function CreateProject({
  isDialogOpen,
  toggleDialog,
  onCreate,
}: {
  isDialogOpen: boolean;
  toggleDialog: () => void;
  onCreate: (name: string) => void;
}) {
  const [dashboardName, setDashboardName] = useState("");
  const [inputError, setInputError] = useState(false);
  const router = useRouter();
  const { toast } = useToast();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.replace(/\s/g, "");
    setDashboardName(newValue);
  };

  const [isNameChecked, setIsNameChecked] = useState(false);
  const [isNameValid, setIsNameValid] = useState(false);
  const handleCheckNameAndContinue = async () => {
    if (dashboardName.trim() === "") {
      setInputError(true);
      return;
    }

    try {
      const data = await checkDashboardName(dashboardName.trim());

      if (data.status) {
        setIsNameValid(true);
        setInputError(false);
        const url = `/dashboardOnboarding?dashboardName=${encodeURIComponent(
          dashboardName.trim()
        )}`;
        router.push(url);
      } else {
        setIsNameValid(false);
        toast({
          title: "Error",
          description:
            "Connection with the same name already exists. Please try another name.",
        });
        setInputError(true);
      }

      setIsNameChecked(true);
    } catch (error) {
      console.error("Error checking connection name:", error);
    }
  };

  return (
    <Dialog open={isDialogOpen} onOpenChange={toggleDialog}>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          onClick={toggleDialog}
          className="text-[#233E7D] hover:text-[#D82026] hover:bg-[#233E7D]/10 text-sm font-medium h-10 px-4 py-2 rounded-md"
        >
          <div className="flex items-center gap-2">
            <Plus /> <span>Create Dashboard</span>
          </div>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[550px] bg-white rounded-md">
        <DialogHeader>
          <DialogTitle className="text-[#233E7D] text-2xl font-bold">New Dashboard</DialogTitle>
          <DialogDescription className="text-base text-[#6B7280]">
            Enter the name of your new Dashboard. Click continue when you&apos;re done.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-1 items-center gap-4">
            <Input
              id="project-name"
              value={dashboardName}
              onChange={handleInputChange}
              className={`w-full border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] ${inputError ? "border-[#D82026]" : ""}`}
              placeholder="Name of Your Dashboard"
            />
            <p className="text-xs text-[#6B7280]">
              Note: Spaces are not allowed. You can use hyphens (-) and underscores (_).
            </p>
          </div>
          {inputError && (
            <p className="text-[#D82026] text-xs">Dashboard name is required</p>
          )}
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={toggleDialog}
            className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 h-10 px-4 py-2 text-sm font-medium rounded-md"
          >
            Cancel
          </Button>
          <Button
            variant="default"
            onClick={handleCheckNameAndContinue}
            className="bg-[#D82026] hover:bg-[#b81a1f] text-white h-10 px-4 py-2 text-sm font-medium rounded-md"
          >
            Continue
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

const cloudPlatformConfig = {
  aws: { name: "AWS", icon: AWS },
  gcp: { name: "Google Cloud", icon: GCP },
  azure: { name: "Azure", icon: AZURE },
  snowflake: { name: "Snowflake", icon: SNOWFLAKE },
};

type CloudPlatformKey = keyof typeof cloudPlatformConfig;

const Row = ({
  row,
  onDelete,
}: {
  row: GroupedDashboard;
  onDelete: (id: number) => void;
}) => {
  const [open, setOpen] = useState(() => {
    const storedState = localStorage.getItem(`dashboard-${row.name}`);
    return storedState === 'true';
  });
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isSyncDialogOpen, setIsSyncDialogOpen] = useState(false);
  const [dashboardToDelete, setDashboardToDelete] = useState<number | null>(null);
  const [dashboardToSync, setDashboardToSync] = useState<{id: number, persona: string} | null>(null);
  const { toast } = useToast();
  const router = useRouter();
  const [showAllConnectors, setShowAllConnectors] = useState(false);

  const [isAddPersonaDialogOpen, setIsAddPersonaDialogOpen] = useState(false);
  const [newPersona, setNewPersona] = useState("");

  const handleAddPersona = async () => {
    if (!newPersona.trim()) {
      toast({
        title: "Error",
        description: "Persona name cannot be empty",
      });
      return;
    }

    try {
      const response = await axiosInstance.post(
        `${BACKEND}/dashboard/add_persona?name=${encodeURIComponent(row.name)}`,
        [newPersona.trim()]
      );

      if (response.status === 200) {
        toast({
          title: "Success",
          description: `New persona "${newPersona}" added successfully.`,
        });
        window.location.reload();
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || "Failed to add new persona.";
      toast({
        title: "Error",
        description: typeof errorMessage === 'string' 
          ? errorMessage 
          : "All provided personas already exist for this dashboard",
        variant: "destructive",
      });
    }

    setIsAddPersonaDialogOpen(false);
    setNewPersona("");
  };

  useEffect(() => {
    localStorage.setItem(`dashboard-${row.name}`, open.toString());
  }, [open, row.name]);

  const handleRowClick = (dashboardId: number, persona: string) => {
    const dashboard = row.dashboards.find(d => d.id === dashboardId);
    if (!dashboard || dashboard.status === false) return;
    router.push(
      `/dashboard-home/${dashboardId}/${row.name}/${encodeURIComponent(persona)}`
    );
  };

  const handleDeleteConfirmation = async () => {
    if (dashboardToDelete === null) return;
    
    try {
      await deleteDashboard(dashboardToDelete);
      onDelete(dashboardToDelete);
      setIsDeleteDialogOpen(false);
      setDashboardToDelete(null);
    } catch (error) {
      console.error("Error deleting dashboard:", error);
    }
  };

  const handleSync = async () => {
    if (!dashboardToSync) return;

    try {
      const response = await axiosInstance.get(`${BACKEND}/dashboard/${dashboardToSync.id}/run_ingestion`, {
        headers: {
          "Content-Type": "application/json",
          accept: "application/json",
        },
      });

      if (response.status === 200) {
        toast({
          title: "Success",
          description: `Dashboard "${row.name}" synced successfully.`,
        });
      } else {
        throw new Error("Failed to sync dashboard");
      }
    } catch (error) {
      console.error("Error syncing dashboard:", error);
      toast({
        title: "Error",
        description: "Failed to sync dashboard.",
      });
    }

    setIsSyncDialogOpen(false);
    setDashboardToSync(null);
  };

  const cloudConfigs = useMemo(() => {
    return row.cloud_platforms
      .map(platform => {
        const normalizedPlatform = platform.toLowerCase() as CloudPlatformKey;
        return cloudPlatformConfig[normalizedPlatform] || {
          name: platform,
          icon: null,
        };
      })
      .filter(config => config);
  }, [row.cloud_platforms]);

  const displayedConnectors = showAllConnectors 
    ? row.connectors 
    : row.connectors.slice(0, 3);

  return (
    <>
      <tr className="border-b border-[#E0E5EF] hover:bg-[#233E7D]/5 transition-colors">
        <td className="p-4">
          <button
            onClick={() => setOpen(!open)}
            className="p-1 hover:bg-[#233E7D]/10 rounded-full transition-colors"
          >
            {open ? <ChevronUp size={20} className="text-[#233E7D]" /> : <ChevronDown size={20} className="text-[#233E7D]" />}
          </button>
        </td>
        <td className="p-4 font-semibold text-[#233E7D] text-base">{row.name}</td>
        <td className="p-4">
          <div className="flex items-center gap-4">
            {cloudConfigs.map((config, index) => (
              <div key={index} className="flex items-center">
                {config.icon && (
                  <Image
                    src={config.icon}
                    width={24}
                    height={24}
                    alt={`${config.name} icon`}
                  />
                )}
              </div>
            ))}
          </div>
        </td>
        <td className="p-4 text-[#233E7D] text-base">{new Date(row.date).toLocaleDateString()}</td>
        <td className="p-4">
          <div className="flex flex-wrap items-center">
            <div className="flex-1 flex flex-wrap items-center">
              {displayedConnectors.map((connector) => (
                <Link
                  key={connector.id}
                  href={`/connections/${connector.id}/${row.cloud_platforms[0]}`}
                >
                  <span className="text-[#233E7D] hover:text-[#D82026] hover:underline mr-2 mb-2 transition-colors text-base">
                    {connector.name}
                  </span>
                </Link>
              ))}
              {row.connectors.length > 3 && !showAllConnectors && (
                <button
                  onClick={() => setShowAllConnectors(true)}
                  className="text-xs text-[#233E7D] hover:text-[#D82026] cursor-pointer underline mr-2"
                >
                  Show More
                </button>
              )}
              {showAllConnectors && row.connectors.length > 3 && (
                <button
                  onClick={() => setShowAllConnectors(false)}
                  className="text-xs text-[#233E7D] hover:text-[#D82026] cursor-pointer underline mr-2"
                >
                  Show Less
                </button>
              )}
            </div>
          </div>
        </td>
      </tr>
      {/* Add new Dialog for adding persona */}
      <Dialog open={isAddPersonaDialogOpen} onOpenChange={setIsAddPersonaDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-[#233E7D]">Add New Persona</DialogTitle>
            <DialogDescription>
              Select a persona to add to dashboard "{row.name}".
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-1 items-center gap-4">
              <Select
                value={newPersona}
                onValueChange={setNewPersona}
              >
                <SelectTrigger className="w-full border-[#233E7D]">
                  <SelectValue placeholder="Select a persona" />
                </SelectTrigger>
                <SelectContent>
                  {PERSONA_OPTIONS.map((persona) => (
                    <SelectItem key={persona.value} value={persona.value}>
                      {persona.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsAddPersonaDialogOpen(false)}
              className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
            >
              Cancel
            </Button>
            <Button
              variant="default"
              onClick={handleAddPersona}
              disabled={!newPersona}
              className="bg-[#D82026] hover:bg-[#b81a1f] text-white"
            >
              Add
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {open && (
        <tr>
          <td colSpan={5}>
            <div className="p-4 pl-20 bg-[#F9FEFF]">
              <table className="w-full">
                <tbody>
                  {row.dashboards.flatMap(dashboard => 
                    dashboard.persona.map((persona, index) => (
                      <tr key={`${dashboard.id}-${index}`} className="border-b last:border-b-0">
                        <td className="p-2">
                          <button
                            onClick={() => handleRowClick(dashboard.id, persona)}
                            disabled={dashboard.status === false}
                            className={`mr-2 ${
                              dashboard.status === false
                                ? "text-gray-400 cursor-not-allowed"
                                : "text-[#233E7D] hover:text-[#D82026] hover:underline transition-colors"
                            }`}
                          >
                            {persona}
                          </button>
                          {dashboard.status === false && (
                            <span className="text-yellow-600 text-xs ml-2">
                              (Data Ingestion in progress)
                            </span>
                          )}
                        </td>
                        {/* <td className="p-2">
                          <button
                            className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                            disabled={dashboard.status === false}
                          >
                            <Megaphone size={20} />
                          </button>
                        </td> */}
                        {/* <td className="p-2">
                          <button
                            className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                            disabled={dashboard.status === false}
                          >
                            <Share2 size={20} />
                          </button>
                        </td> */}
                        <td className="p-2">
                          <button
                            onClick={() => {
                              setDashboardToDelete(dashboard.id);
                              setIsDeleteDialogOpen(true);
                            }}
                            disabled={dashboard.status === false}
                            className={`p-1 rounded-full transition-colors group relative ${
                              dashboard.status === false
                                ? "hover:bg-gray-100"
                                : "hover:bg-[#D82026]/10"
                            }`}
                          >
                            <Trash2 size={20} color="#D82026" />
                            <span className="invisible group-hover:visible absolute -top-8 left-1/2 transform -translate-x-1/2 
                              px-2 py-1 bg-gray-800 text-white text-xs rounded whitespace-nowrap">
                              Delete Dashboard
                            </span>
                          </button>
                        </td>
                        <td className="p-2">
                          <button
                            onClick={() => {
                              setDashboardToSync({id: dashboard.id, persona});
                              setIsSyncDialogOpen(true);
                            }}
                            disabled={dashboard.status === false}
                            className={`p-1 rounded-full transition-colors group relative ${
                              dashboard.status === false
                                ? "hover:bg-gray-100"
                                : "hover:bg-[#233E7D]/10"
                            }`}
                          >
                            <RefreshCcw size={20} color="#233E7D" />
                            <span className="invisible group-hover:visible absolute -top-8 left-1/2 transform -translate-x-1/2 
                              px-2 py-1 bg-gray-800 text-white text-xs rounded whitespace-nowrap">
                              Sync Dashboard
                            </span>
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                  <tr className="border-b last:border-b-0">
                    <td colSpan={4} className="p-2">
                      <Button
                        variant="ghost"
                        onClick={() => setIsAddPersonaDialogOpen(true)}
                        className="flex items-center gap-2 text-[#233E7D] hover:text-[#D82026] hover:bg-[#233E7D]/10 rounded-sm px-0"
                      >
                        <Plus className="h-4 w-4" />
                        <span className="text-sm">Add new persona</span>
                      </Button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </td>
        </tr>
      )}

      {/* Delete Dialog */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-[#D82026]">Confirm Delete</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this dashboard persona? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsDeleteDialogOpen(false)}
              className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirmation}
              className="bg-[#D82026] hover:bg-[#b81a1f] text-white"
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Sync Dialog */}
      <Dialog open={isSyncDialogOpen} onOpenChange={setIsSyncDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-[#233E7D]">Confirm Sync</DialogTitle>
            <DialogDescription>
              Are you sure you want to sync this dashboard persona? This action will update the dashboard with the latest data.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsSyncDialogOpen(false)}
              className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
            >
              Cancel
            </Button>
            <Button
              variant="default"
              onClick={handleSync}
              className="bg-[#233E7D] hover:bg-[#19294e] text-white"
            >
              Sync
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

const DashboardPage = () => {
  const [dashboards, setDashboards] = useState<Dashboard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  useEffect(() => {
    const getDashboards = async () => {
      try {
        setLoading(true);
        const fetchedDashboards = await fetchDashboards();
        setDashboards(fetchedDashboards);
        setError(null);
      } catch (error) {
        console.error("Error fetching dashboards:", error);
        setError("Failed to load dashboards. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    getDashboards();
  }, []);

  // Group dashboards by name
  const groupedRows = useMemo(() => {
    const groups: { [name: string]: GroupedDashboard } = {};
    
    dashboards.forEach(dashboard => {
      if (!groups[dashboard.name]) {
        // Create new group
        groups[dashboard.name] = {
          name: dashboard.name,
          dashboards: [dashboard],
          cloud_platforms: [...dashboard.cloud_platforms],
          personas: [...dashboard.persona],
          connectors: [...dashboard.connectors],
          date: dashboard.date
        };
      } else {
        // Add to existing group
        const group = groups[dashboard.name];
        group.dashboards.push(dashboard);
        
        // Merge cloud platforms
        dashboard.cloud_platforms.forEach(platform => {
          if (!group.cloud_platforms.includes(platform)) {
            group.cloud_platforms.push(platform);
          }
        });
        
        // Merge personas
        dashboard.persona.forEach(persona => {
          if (!group.personas.includes(persona)) {
            group.personas.push(persona);
          }
        });
        
        // Merge connectors
        dashboard.connectors.forEach(connector => {
          if (!group.connectors.some(c => c.id === connector.id)) {
            group.connectors.push(connector);
          }
        });
        
        // Use latest date
        const currentDate = new Date(group.date);
        const newDate = new Date(dashboard.date);
        if (newDate > currentDate) {
          group.date = dashboard.date;
        }
      }
    });
    
    return Object.values(groups);
  }, [dashboards]);

  const handleDelete = (id: number) => {
    setDashboards(prev => prev.filter(dashboard => dashboard.id !== id));
  };

  return (
  <div className="p-4 max-w-8xl mx-auto bg-[#F9FEFF]">
    <h1 className="p-4 text-3xl font-bold mb-6 text-[#233E7D]">Dashboard</h1>
    <div className="overflow-x-auto shadow-md rounded-lg">
      <table className="w-full border-collapse bg-white rounded-md">
        <thead>
          <tr className="bg-[#233E7D]">
            <th className="p-4"></th>
            <th className="p-4 text-left font-semibold text-white text-base">Name</th>
            <th className="p-4 text-left font-semibold text-white text-base">Cloud Platform</th>
            <th className="p-4 text-left font-semibold text-white text-base">Creation Date</th>
            <th className="p-4 text-left font-semibold text-white text-base">Connectors</th>
          </tr>
        </thead>
        <tbody>
          {groupedRows.map(row => (
            <Row key={row.name} row={row} onDelete={handleDelete} />
          ))}
        </tbody>
      </table>
      <div className="p-4">
        <CreateProject
          isDialogOpen={isDialogOpen}
          toggleDialog={() => setIsDialogOpen(!isDialogOpen)}
          onCreate={(name) => {
            console.log("Create dashboard:", name);
          }}
        />
      </div>
    </div>
  </div>
);
};

export default DashboardPage;
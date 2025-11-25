"use client";
import { Card, CardContent } from "@/components/ui/card";
import React, { useState, useMemo, useEffect } from "react";
import { useParams } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import axiosInstance, { BACKEND, fetchProjectDetails } from "@/lib/api";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { PlusCircle, ArrowUpDown, Trash2, Plus, RefreshCcw, FilterX, X, TagIcon } from "lucide-react";
import Loader from "@/components/Loader";
import { ChevronRight, ChevronLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import NewAlertRuleForm from "@/components/alertRules/NewAlertRuleForm"; // Ensure you import your form component



type Tag = {
  tag_id: number;
  key: string;
  value: string;
};

type Resource = {
  id: number;
  resource_name: string;
  resource_id: string; 
  region_id: string;
  region_name: string;
  service_category: string;
  service_name: string;
  resource_group_name: string;
  tags: Tag[];
};

type Filters = {
  service_category: string;
  service_name: string;
  region_name: string;
  resource_group_name: string;
  tag: string;
};

type FilterOptions = {
  service_categories: string[];
  service_names: string[];
  region_names: string[];
  resource_group_names: string[];
  tags: string[];
};
const handleNewAlertClick = () => {
  console.log(" button clicked");
};

const tagColors: { [key: string]: string } = {
  Environment: "bg-blue-100 text-blue-800",
  CostCenter: "bg-green-100 text-green-800",
  Project: "bg-purple-100 text-purple-800",
  Application: "bg-yellow-100 text-yellow-800",
};

const getTagColor = (key: string) => {
  return tagColors[key] || "bg-gray-100 text-gray-800";
};

interface FilterSelectProps {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  onClear: () => void;
}

const FilterSelect: React.FC<FilterSelectProps> = ({ label, value, options, onChange, onClear }) => (
  <div className="relative">
    {value === "all" ? (
      <Select onValueChange={onChange} value={value}>
        <SelectTrigger className="w-[180px]" placeholder={label}>
          <SelectValue placeholder={label} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All {label}s</SelectItem>
          {options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    ) : (
      <div className="flex items-center w-[180px] h-10 px-3 py-2 text-sm border rounded-md">
        <span className="flex-grow truncate">{value}</span>
        <Button
          variant="ghost"
          size="sm"
          className="p-0 h-4 w-4"
          onClick={onClear}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )}
  </div>
);

export default function ResourceTable() {
  const [isButtonVisible, setIsButtonVisible] = useState(false);
  const [showNewAlertForm, setShowNewAlertForm] = useState(false);
  const [resources, setResources] = useState<Resource[]>([]);
  const [filters, setFilters] = useState<Filters>({
    service_category: "all",
    service_name: "all",
    region_name: "all",
    resource_group_name: "all",
    tag: "all",
  });
  const handleNewAlertClick = () => {
    setShowNewAlertForm(true);
    setIsButtonVisible(false); // Add this line to hide the button when clicked
  };
  const [projectName, setProjectName] = useState<string | null>(null);
  const params = useParams();
  const projectId = params.projectName as string;
  const [sortConfig, setSortConfig] = useState<{
    key: keyof Resource;
    direction: "asc" | "desc";
  } | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectAll, setSelectAll] = useState(false);
  const [selectedResourceIds, setSelectedResourceIds] = useState<number[]>([]);  const [isTagDialogOpen, setIsTagDialogOpen] = useState(false);
  const [newTagKey, setNewTagKey] = useState<string | null>(null);
  const [newTagValue, setNewTagValue] = useState<string | null>(null);
  const [selectedTagId, setSelectedTagId] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [currentPageRowCount, setCurrentPageRowCount] = useState(0);
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    service_categories: [],
    service_names: [],
    region_names: [],
    resource_group_names: [],
    tags: [],
  });
  const [isSyncing, setIsSyncing] = useState(false);
  const [syncStatus, setSyncStatus] = useState<string | null>(null);
  const [availableTags, setAvailableTags] = useState<Tag[]>([]);
  const [pageSize, setPageSize] = useState(10);
  const [loading, setLoading] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const cloudPlatform = params.cloudPlatform as string;
const [allResourceIds, setAllResourceIds] = useState<number[]>([]);
const [isLoadingAllResources, setIsLoadingAllResources] = useState(false);
const [hasFetchedAllIds, setHasFetchedAllIds] = useState(false);
  const router = useRouter();

    // Add a new memoized value for selected resource names
    const selectedResourceNames = useMemo(() => {
      if (!selectedResourceIds || !resources) return '';
      
      return resources
        .filter(resource => selectedResourceIds.includes(resource.id))
        .map(resource => resource.resource_name)
        .join(',');
    }, [selectedResourceIds, resources]);

  const selectedResourceNamesArray = selectedResourceNames.split(',');
 
  useEffect(() => {
    const fetchProjectName = async () => {
      try {
        const projectDetails = await fetchProjectDetails(projectId);
        setProjectName(projectDetails.name);
      } catch (error) {
        console.error("Error fetching project name:", error);
      }
    };
    fetchProjectName();
  }, [projectId]);

  useEffect(() => {
    const fetchResources = async () => {
      setLoading(true);
      let attempts = 0;
      const maxAttempts = 3;
      
      const makeRequest = async () => {
        try {
          if (projectName) {
            const params = {
              name: projectName,
              page: currentPage.toString(),
              page_size: pageSize.toString(),
              service_category: filters.service_category !== "all" ? filters.service_category : "",
              service_name: filters.service_name !== "all" ? filters.service_name : "",
              resource_group_name: filters.resource_group_name !== "all" ? filters.resource_group_name : "",
              region_name: filters.region_name !== "all" ? filters.region_name : "",
              tag: filters.tag !== "all" ? filters.tag : "",
              sort_by: sortConfig ? sortConfig.key : "resource_name",
              sort_order: sortConfig ? sortConfig.direction : "asc",
            };
  
            const response = await axiosInstance.get(`${BACKEND}/resources/resources`, { params });
            const data = response.data;
            const fetchedResources = data.resources || [];
            setResources(fetchedResources);
            console.log("fetchedResources", fetchedResources);
            
            const totalItems = data.total_items || 0;
            const calculatedTotalPages = Math.ceil(totalItems / pageSize);
            setTotalItems(totalItems);
            setTotalPages(calculatedTotalPages > 0 ? calculatedTotalPages : 1);
            
            setFilterOptions({
              service_categories: data.filter_options.service_categories || [],
              service_names: data.filter_options.service_names || [],
              region_names: data.filter_options.region_names || [],
              resource_group_names: data.filter_options.resource_group_name || [],
              tags: data.filter_options.tags || [],
            });
            
            const currentPageRowCount = fetchedResources.length;
            setCurrentPageRowCount(currentPageRowCount);
            console.log(`Total items: ${totalItems}`);
            console.log(`Number of rows in current page: ${currentPageRowCount}`);
            setLoading(false);
          }
        } catch (error) {
          attempts++;
          if (attempts < maxAttempts) {
            setRetryCount(attempts);
            makeRequest();
          } else {
            console.error("Failed to fetch resources after multiple attempts:", error);
            setLoading(false);
          }
        }
      };
      
      makeRequest();
    };
    
    fetchResources();
  }, [projectName, currentPage, pageSize, filters, sortConfig]);

  useEffect(() => {
    const fetchTags = async () => {
      try {
        const response = await axiosInstance.get(`${BACKEND}/tags/tags`);
        setAvailableTags(response.data);  // Axios stores the response data in the `data` field
      } catch (error) {
        console.error("Error fetching tags:", error);
      }
    };
    
    fetchTags();
  }, []);

  const handleFilterChange = (key: keyof Filters, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const clearFilter = (key: keyof Filters) => {
    setFilters((prev) => ({ ...prev, [key]: "all" }));
  };

  const handleSort = (key: keyof Resource) => {
    setSortConfig((prev) =>
      prev && prev.key === key
        ? { ...prev, direction: prev.direction === "asc" ? "desc" : "asc" }
        : { key, direction: "asc" }
    );
  };

  const filteredAndSortedResources = useMemo(() => {
    if (!resources || resources.length === 0) return [];
    return resources
      .filter(
        (resource) =>
          (filters.service_category === "all" ||
            resource.service_category === filters.service_category) &&
          (filters.service_name === "all" ||
            resource.service_name === filters.service_name) &&
          (filters.region_name === "all" ||
            resource.region_name === filters.region_name) &&
          (filters.resource_group_name === "all" ||
            resource.resource_group_name === filters.resource_group_name) &&
          (
            filters.tag === "all" ||
            (
              filters.tag === "untagged"
                ? resource.tags.length === 0
                : resource.tags.some(
                    (tag) =>
                      `${tag.key}:${tag.value}`.toLowerCase() === filters.tag.toLowerCase()
                  )
            )
          ) &&
          (searchTerm === "" ||
            Object.values(resource).some(
              (value) =>
                typeof value === "string" &&
                value.toLowerCase().includes(searchTerm.toLowerCase())
            ))
      )
      .sort((a, b) => {
        if (!sortConfig) return 0;
        const { key, direction } = sortConfig;
        if (a[key] < b[key]) return direction === "asc" ? -1 : 1;
        if (a[key] > b[key]) return direction === "asc" ? 1 : -1;
        return 0;
      });
  }, [resources, filters, sortConfig, searchTerm]);

const [isCreatingNewTag, setIsCreatingNewTag] = useState(false);
const [newTagKeyInput, setNewTagKeyInput] = useState("");
const [newTagValueInput, setNewTagValueInput] = useState("");
const [newTagBudgetInput, setNewTagBudgetInput] = useState("");
const [errorMessage, setErrorMessage] = useState<string | null>(null);

const handleCreateNewTag = async () => {
  try {
    const response = await axiosInstance.post(`${BACKEND}/tags/tags`, {
      key: newTagKeyInput,
      value: newTagValueInput,
      budget: newTagBudgetInput,
    });

    if (response.status === 200) {
      setAvailableTags(prev => [...prev, response.data]);
      setNewTagKey(newTagKeyInput);
      setNewTagValue(newTagValueInput);
      setSelectedTagId(response.data.tag_id);
      setIsCreatingNewTag(false);
      setNewTagKeyInput("");
      setNewTagValueInput("");
      setNewTagBudgetInput("");
      setErrorMessage(null);
    }
  } catch (error: any) {
    const errorMsg = error.response?.data?.detail || "Failed to create tag";
    setErrorMessage(errorMsg);
  }
};



// Optimized select all: fetch all IDs only once and cache them

useEffect(() => {
  if (selectAll && !hasFetchedAllIds && projectName) {
    const fetchAllResourceIds = async () => {
      setIsLoadingAllResources(true);
      try {
        // Try to fetch only IDs for better performance
        let ids: number[] = [];
        try {
          // If your backend supports a lightweight endpoint for just IDs, use it here
          const idResponse = await axiosInstance.get(`${BACKEND}/resources/resource-ids`, {
            params: {
              name: projectName,
              service_category: filters.service_category !== "all" ? filters.service_category : "",
              service_name: filters.service_name !== "all" ? filters.service_name : "",
              resource_group_name: filters.resource_group_name !== "all" ? filters.resource_group_name : "",
              region_name: filters.region_name !== "all" ? filters.region_name : "",
              tag: filters.tag !== "all" ? filters.tag : "",
            }
          });
          ids = idResponse.data.ids || [];
        } catch (idErr) {
          // Fallback to fetching all resources but only map IDs
          const response = await axiosInstance.get(`${BACKEND}/resources/resources`, {
            params: {
              name: projectName,
              service_category: filters.service_category !== "all" ? filters.service_category : "",
              service_name: filters.service_name !== "all" ? filters.service_name : "",
              resource_group_name: filters.resource_group_name !== "all" ? filters.resource_group_name : "",
              region_name: filters.region_name !== "all" ? filters.region_name : "",
              tag: filters.tag !== "all" ? filters.tag : "",
              page_size: totalItems > 0 ? totalItems : 10000,
              page: 1,
              fields: "id", // If backend supports field selection
            }
          });
          ids = (response.data.resources || []).map((resource: Resource) => resource.id);
        }
        setAllResourceIds(ids);
        setSelectedResourceIds(ids);
        setHasFetchedAllIds(true);
      } catch (error) {
        console.error("Error fetching all resource IDs:", error);
        // Fallback to current page
        const currentPageIds = resources.map((resource: Resource) => resource.id);
        setSelectedResourceIds(currentPageIds);
      } finally {
        setIsLoadingAllResources(false);
      }
    };
    fetchAllResourceIds();
  } else if (!selectAll) {
    setHasFetchedAllIds(false);
  }
}, [selectAll, hasFetchedAllIds, projectName, filters, totalItems, resources]);

const handleSelectAll = () => {
  if (!selectAll) {
    if (allResourceIds.length > 0) {
      setSelectedResourceIds(allResourceIds);
      setHasFetchedAllIds(true);
    } 
  } else {
    setSelectedResourceIds([]);
    setHasFetchedAllIds(false);
  }
  setSelectAll((prev) => !prev);
};

const handleResourceSelect = (resourceId: number) => {
  setSelectedResourceIds(prev => {
    const newSelection = prev.includes(resourceId)
      ? prev.filter(id => id !== resourceId)
      : [...prev, resourceId];
    
    if (allResourceIds.length > 0) {
      setSelectAll(newSelection.length === allResourceIds.length);
    } else {
      setSelectAll(newSelection.length === resources.length);
    }
    
    return newSelection;
  });
};

  const handleBulkTagging = () => {
    setIsTagDialogOpen(true);
    setNewTagKey(null);
    setNewTagValue(null);
    setSelectedTagId(null);
  };

  const availableValuesForKey = useMemo(() => {
    if (!Array.isArray(availableTags)) {
      console.error('availableTags is not an array:', availableTags);
      return [];
    }
    return availableTags
      .filter((tag) => tag.key === newTagKey)
      .map((tag) => ({ value: tag.value, tag_id: tag.tag_id }));
  }, [newTagKey, availableTags]);

  const applyNewTag = async () => {
    if (selectedTagId && newTagKey && newTagValue) {
      try {
        const response = await axiosInstance.post(`${BACKEND}/resources/apply-tag`, {
          tag_id: selectedTagId,
          resource_ids: selectedResourceIds,
        });
  
        const { resource_status } = response.data;
  
        setResources((prevResources: Resource[]) =>
          prevResources.map((resource: Resource) => {
            const statusItem = resource_status.find(
              (res: { id: number }) => res.id === resource.id
            );
  
            if (statusItem && (statusItem.status === "success" || statusItem.status === "already_applied")) {
              const tagExists = resource.tags.some(
                (tag) => tag.tag_id === selectedTagId
              );
  
              if (!tagExists) {
                return {
                  ...resource,
                  tags: [
                    ...resource.tags,
                    { tag_id: selectedTagId, key: newTagKey, value: newTagValue },
                  ],
                };
              }
            }
  
            return resource;
          })
        );
  
        const failedResources = resource_status.filter(
          (resource: { status: string }) => resource.status === "failed"
        );
  
        if (failedResources.length > 0) {
          console.error("Failed to apply tag to resources:", failedResources);
        }
      } catch (error) {
        console.error("Error applying tag:", error);
      } finally {
        setIsTagDialogOpen(false);
        setSelectedResourceIds([]);
        setSelectAll(false);
      }
    }
  };

  const handleDeleteTag = async (id: number, tagId: number) => {
    try {
      const response = await axiosInstance.delete(`${BACKEND}/resources/remove-tag`, {
        params: {
          id: id,
          tag_id: tagId
        }
      });
      
      if (response.status === 200) {
        setResources((prevResources) =>
          prevResources.map((resource) =>
            resource.id === id
              ? {
                  ...resource,
                  tags: resource.tags.filter((tag) => tag.tag_id !== tagId),
                }
              : resource
          )
        );
      } else {
        console.error("Failed to delete tag");
      }
    } catch (error) {
      console.error("Error deleting tag:", error);
    }
  };
  const handleSyncResources = async () => {
    setIsSyncing(true);
    setSyncStatus(null);
    try {
      const response = await axiosInstance.post(
        `${BACKEND}/resources/sync-resources`,
        null,
        {
          params: {
            schema: projectName,
            cloudPlatform: cloudPlatform
          }
        }
      );
      const data = response.data;
      if (data.status) {
        setSyncStatus("Resources synced successfully!");
      } else {
        setSyncStatus(`Failed to sync resources: ${data.message}`);
      }
    } catch (error) {
      console.error("Error syncing resources:", error);
      setSyncStatus("Failed to sync resources.");
    }
    setIsSyncing(false);
    setTimeout(() => {
      setSyncStatus(null);
    }, 3000);
  };
const clearAllFilters = () => {
 setFilters({
   service_category: "all",
   service_name: "all",
   region_name: "all",
   resource_group_name: "all",
   tag: "all",
 });
 setSearchTerm("");
};
const SyncingSpinner = () => (
 <div className="w-4 h-4 border-2 border-t-2 border-white-500 border-solid rounded-full animate-spin"></div>
);

const handleBackClick = () => {
  setShowNewAlertForm(false);
};
// const handleBackClick = () => {
//   setShowNewAlertForm(false);
//   // Navigate back to AWS resources page
//   router.push(`/connections/${projectId}/aws`);
// };

return (
<div className="container mx-auto p-4 bg-[#F9FEFF] rounded-lg min-h-screen">
    <div className="flex justify-between items-center mb-4">
      <h1 className="text-2xl font-bold text-[#233E7D]">Resource View</h1>
      <div className="flex justify-center items-center gap-4">
        <Button
          onClick={handleBulkTagging}
          disabled={selectedResourceIds.length === 0}
          className="bg-[#233E7D] text-white hover:bg-[#19294e] text-sm font-medium rounded-md h-10 px-4 py-2"
        >
          <TagIcon className="h-4 w-4 mr-2 text-white" /> Apply Tag
        </Button>
        <Button
          onClick={handleNewAlertClick}
          disabled={selectedResourceIds.length === 0}
          className="bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md h-10 px-4 py-2"
        >
          <Plus className="h-4 w-4 text-white" />
          New Alert Rule
        </Button>
        <Button
          onClick={handleSyncResources}
          className="bg-[#233E7D] text-white hover:bg-[#19294e] text-sm font-medium rounded-md h-10 px-4 py-2"
          disabled={isSyncing}
        >
          {isSyncing ? (
            <SyncingSpinner />
          ) : (
            <RefreshCcw className="h-4 w-4 text-white" />
          )}
        </Button>
      </div>
    </div>
          {/* Add Selected Resources Display after the header
          {selectedResourceIds.length > 0 && (
        <Card className="mb-4">
          <CardContent className="pt-6">
            <div className="bg-gray-50 p-4 rounded-md">
              <pre className="whitespace-pre-wrap break-all font-mono text-sm">
                "{selectedResourceNames}"
              </pre>
            </div>
          </CardContent>
        </Card>
      )} */}
    {syncStatus && (
      <div
        className={`mb-4 p-2 ${
          syncStatus.includes("Failed")
            ? "bg-red-100 text-red-800"
            : "bg-green-100 text-green-800"
        } border border-solid border-current rounded`}
      >
        {syncStatus}
      </div>
    )}
    <div className="flex gap-4 mb-4">
      <Input
        placeholder="Search resources..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="max-w-sm"
      />
      <FilterSelect
        label="Service Category"
        value={filters.service_category}
        options={filterOptions.service_categories}
        onChange={(value) => handleFilterChange("service_category", value)}
        onClear={() => clearFilter("service_category")}
      />
      <FilterSelect
        label="Service Name"
        value={filters.service_name}
        options={filterOptions.service_names}
        onChange={(value) => handleFilterChange("service_name", value)}
        onClear={() => clearFilter("service_name")}
      />
      {cloudPlatform !== 'aws' && (
        <FilterSelect
          label="Resource Group Name"
          value={filters.resource_group_name}
          options={filterOptions.resource_group_names}
          onChange={(value) => handleFilterChange("resource_group_name", value)}
          onClear={() => clearFilter("resource_group_name")}
        />
      )}
      <FilterSelect
        label="Region"
        value={filters.region_name}
        options={filterOptions.region_names}
        onChange={(value) => handleFilterChange("region_name", value)}
        onClear={() => clearFilter("region_name")}
      />
      <FilterSelect
        label="Tag"
        value={filters.tag}
        options={[
          "untagged",
          ...availableTags.map((tag) => `${tag.key}:${tag.value}`)
        ]}
        onChange={(value) => handleFilterChange("tag", value)}
        onClear={() => clearFilter("tag")}
      />
      <Button
        onClick={clearAllFilters}
        className="bg-[#233E7D] text-white hover:bg-[#19294e] text-sm font-medium rounded-md h-10 px-4 py-2"
      >
        <FilterX className="h-4 w-4 text-white"/>
      </Button>
   </div>
   <Table className="bg-white border border-[#E0E5EF] rounded-md shadow-md">
  <TableHeader>
    <TableRow>
      <TableHead className="w-[50px]">
        <Checkbox 
          checked={selectAll || (allResourceIds.length > 0 && selectedResourceIds.length === allResourceIds.length)} 
          onCheckedChange={handleSelectAll}
          disabled={isLoadingAllResources}
        />    
      </TableHead>
      {cloudPlatform === 'aws' ? (
        <TableHead
          className="cursor-pointer"
          onClick={() => handleSort("resource_id")}
        >
          Resource ID{" "}
          {sortConfig?.key === "resource_id" && (
            <ArrowUpDown className="ml-2 h-4 w-4 inline" />
          )}
        </TableHead>
      ) : (
        <TableHead
          className="cursor-pointer"
          onClick={() => handleSort("resource_name")}
        >
          Resource Name{" "}
          {sortConfig?.key === "resource_name" && (
            <ArrowUpDown className="ml-2 h-4 w-4 inline" />
          )}
        </TableHead>
      )}
      <TableHead
        className="cursor-pointer"
        onClick={() => handleSort("service_category")}
      >
        Service Category{" "}
        {sortConfig?.key === "service_category" && (
          <ArrowUpDown className="ml-2 h-4 w-4 inline" />
        )}
      </TableHead>
      <TableHead
        className="cursor-pointer"
        onClick={() => handleSort("service_name")}
      >
        Service Name{" "}
        {sortConfig?.key === "service_name" && (
          <ArrowUpDown className="ml-2 h-4 w-4 inline" />
        )}
      </TableHead>
      {cloudPlatform !== 'aws' && (
        <TableHead
          className="cursor-pointer"
          onClick={() => handleSort("resource_group_name")}
        >
          Resource Group Name{" "}
          {sortConfig?.key === "resource_group_name" && (
            <ArrowUpDown className="ml-2 h-4 w-4 inline" />
          )}
        </TableHead>
      )}
      <TableHead
        className="cursor-pointer"
        onClick={() => handleSort("region_name")}
      >
        Region{" "}
        {sortConfig?.key === "region_name" && (
          <ArrowUpDown className="ml-2 h-4 w-4 inline" />
        )}
      </TableHead>
      <TableHead>Tags</TableHead>
    </TableRow>
  </TableHeader>
  <TableBody>
    {loading ? (
      <TableRow>
        <TableCell colSpan={cloudPlatform === 'aws' ? 6 : 6} className="text-center py-8">
          <div className="h-full w-full flex items-center justify-center">
            <Loader />
          </div>
        </TableCell>
      </TableRow>
    ) : (
      filteredAndSortedResources.map((resource) => (
        <TableRow key={resource.id} className="hover:border-[#233E7D] hover:bg-[#233E7D]/5 transition-colors">
          <TableCell>
            <Checkbox
              checked={selectedResourceIds.includes(resource.id)}
              onCheckedChange={() => handleResourceSelect(resource.id)}
            />
          </TableCell>
          {cloudPlatform === 'aws' ? (
            <TableCell className="font-medium">{resource.resource_id}</TableCell>
          ) : (
            <TableCell className="font-medium">{resource.resource_name}</TableCell>
          )}
          <TableCell>{resource.service_category}</TableCell>
          <TableCell>{resource.service_name}</TableCell>
          {cloudPlatform !== 'aws' && (
            <TableCell>{resource.resource_group_name}</TableCell>
          )}
          <TableCell>{resource.region_name}</TableCell>
          <TableCell>
            <div className="flex flex-wrap gap-2">
              {resource.tags.length > 0 ? (
                resource.tags.map((tag) => (
                  <span
                    key={tag.tag_id}
                    className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getTagColor(tag.key)}`}
                  >
                    {tag.key}:{tag.value}
                    <button
                      onClick={() => handleDeleteTag(resource.id, tag.tag_id)}
                      className="ml-1 text-gray-500 hover:text-gray-700"
                      title="Delete tag"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </span>
                ))
              ) : (
                <span className="text-gray-500 font-sm"></span>
              )}
              <button
                onClick={() => {
                  setSelectedResourceIds([resource.id]);
                  handleBulkTagging();
                }}
                className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-[#233E7D]/10 text-[#233E7D] hover:bg-[#233E7D]/20"
              >
                <Plus className="h-3 w-3 mr-1 text-[#233E7D]" />
                Add Tag
              </button>
            </div>
          </TableCell>
        </TableRow>
      ))
    )}
  </TableBody>
</Table>
   {/* <div className="flex justify-center items-center mt-4"> */}
     <div className="flex justify-between items-center mt-4">
     <div className="flex flex-col w-full items-center mt-4">
       <div className="flex items-center gap-2">
         <button
           onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
           disabled={currentPage === 1}
           className={`px-5 py-1.5 border border-[#233E7D] rounded-lg bg-white text-[#233E7D] font-semibold shadow-sm hover:shadow-md transition-all duration-150 hover:bg-[#f3f7fd] hover:border-[#19294e] focus:outline-none focus:ring-2 focus:ring-[#233E7D] disabled:bg-[#f3f7fd] disabled:text-[#b3c2e6] disabled:border-[#e0e5ef] disabled:shadow-none disabled:cursor-not-allowed`}
         >
           Previous
         </button>
         <span className="text-sm mx-2">
           Page {currentPage} of {totalPages}
         </span>
         <button
           onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
           disabled={currentPage === totalPages}
           className={`px-5 py-1.5 border border-[#233E7D] rounded-lg bg-white text-[#233E7D] font-semibold shadow-sm hover:shadow-md transition-all duration-150 hover:bg-[#f3f7fd] hover:border-[#19294e] focus:outline-none focus:ring-2 focus:ring-[#233E7D] disabled:bg-[#f3f7fd] disabled:text-[#b3c2e6] disabled:border-[#e0e5ef] disabled:shadow-none disabled:cursor-not-allowed`}
         >
           Next
         </button>
       </div>
     </div>
     <div>
<Select
       onValueChange={(value) => {
         setPageSize(Number(value));
         setCurrentPage(1);  // Reset to page 1 when changing page size
       }}
     >
       <SelectTrigger className="w-[120px]" placeholder={`${pageSize} per page`}>
         <SelectValue placeholder={`${pageSize} per page`} />
       </SelectTrigger>
       <SelectContent>
         <SelectItem value="10">10</SelectItem>
         <SelectItem value="20">20</SelectItem>
         <SelectItem value="50">50</SelectItem>
       </SelectContent>
     </Select>
     </div>
     </div>
   {/* </div> */}
   <Dialog open={isTagDialogOpen} onOpenChange={setIsTagDialogOpen}>
    <DialogContent>
      <DialogHeader>
        <DialogTitle>
          {isCreatingNewTag ? "Create New Tag" : "Apply Tag"}
        </DialogTitle>
      </DialogHeader>
      {errorMessage && (
        <div className="text-red-500 mb-4">
          {errorMessage}
        </div>
      )}
      <div className="grid gap-4 py-4">
        {isCreatingNewTag ? (
          <div className="grid gap-4">
            <div>
              <Label htmlFor="newTagKey">Key</Label>
              <Input
                id="newTagKey"
                type="text"
                value={newTagKeyInput}
                onChange={(e) => setNewTagKeyInput(e.target.value)}
                placeholder="Enter key"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="newTagValue">Value</Label>
              <Input
                id="newTagValue"
                type="text"
                value={newTagValueInput}
                onChange={(e) => setNewTagValueInput(e.target.value)}
                placeholder="Enter value"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="newTagBudget">Yearly Budget</Label>
              <Input
                id="newTagBudget"
                type="number"
                value={newTagBudgetInput}
                onChange={(e) => setNewTagBudgetInput(e.target.value)}
                placeholder="Enter yearly budget"
                className="mt-1.5"
              />
            </div>
          </div>
        ) : (
          <>
            <div>
              <Label htmlFor="tagKey">Key</Label>
              <Select
                value={newTagKey || ""}
                onValueChange={(value) => setNewTagKey(value)}
              >
                <SelectTrigger className="mt-1.5">
                  <SelectValue placeholder="Select a key" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from(new Set(availableTags.map((tag) => tag.key))).map(
                    (key) => (
                      <SelectItem key={key} value={key}>
                        {key}
                      </SelectItem>
                    )
                  )}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="tagValue">Value</Label>
              <Select
                value={newTagValue || ""}
                onValueChange={(value) => {
                  setNewTagValue(value);
                  const selectedTag = availableTags.find(
                    (tag) => tag.key === newTagKey && tag.value === value
                  );
                  if (selectedTag) setSelectedTagId(selectedTag.tag_id);
                }}
                disabled={!newTagKey}
              >
                <SelectTrigger className="mt-1.5">
                  <SelectValue placeholder="Select a value" />
                </SelectTrigger>
                <SelectContent>
                  {availableValuesForKey.map((valueObj) => (
                    <SelectItem key={valueObj.tag_id} value={valueObj.value}>
                      {valueObj.value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </>
        )}
      </div>
      <DialogFooter>
        {isCreatingNewTag ? (
          <div className="flex justify-between w-full">
            <Button
              variant="outline"
              onClick={() => {
                setIsCreatingNewTag(false);
                setErrorMessage(null);
              }}
            >
              Back to Existing Tags
            </Button>
            <Button
              onClick={handleCreateNewTag}
              disabled={!newTagKeyInput || !newTagValueInput}
              className="bg-blue-800 hover:bg-blue-900 text-white"
            >
              Create Tag
            </Button>
          </div>
        ) : (
          <div className="flex justify-between w-full">
            <Button
              variant="outline"
              onClick={() => {
                setIsCreatingNewTag(true);
                setErrorMessage(null);
              }}
            >
              Create New Tag
            </Button>
            <Button 
              onClick={applyNewTag}
              className="bg-blue-800 hover:bg-blue-900 text-white"
              disabled={!selectedTagId}
            >
              Apply Tag
            </Button>
          </div>
        )}
      </DialogFooter>
    </DialogContent>
  </Dialog>

   <div>
   {showNewAlertForm ? (
  <Dialog open={showNewAlertForm} onOpenChange={setShowNewAlertForm}>
    <DialogContent className="w-full max-w-7xl">
      <DialogHeader>
        <DialogTitle>Create New Alert Rule</DialogTitle>
      </DialogHeader>
      <NewAlertRuleForm
        onBack={handleBackClick}
        onSave={handleBackClick}
        projectId={projectId}
        selectedResourceNames={selectedResourceNamesArray}
      />
    </DialogContent>
  </Dialog>
) : (
  // Fallback UI or any other content when the form isn't shown
  <div> 
    {/* Example fallback content */}
    {/* <p>Other content goes here.</p> */}
  </div>
)}
  </div>

 </div>
 )}

'use client'

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";
import { useRouter, useParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Trash2, Edit, Search, Plus } from "lucide-react";
import NewAlertRuleForm from "@/components/alertRules/NewAlertRuleForm";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import Loader from "@/components/Loader";
import axiosInstance, { BACKEND } from "@/lib/api";

type Tag = {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
};

export default function TagManagement() {
  const [tags, setTags] = useState<Tag[]>([]);
  const [selectedTags, setSelectedTags] = useState<number[]>([]);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [showNewAlertForm, setShowNewAlertForm] = useState(false);
  const params = useParams();
  const [editingTag, setEditingTag] = useState<Tag | null>(null);
  const [isButtonVisible, setIsButtonVisible] = useState(true);
  const { toast } = useToast();
  const projectId = params.projectName as string;
  const [newTag, setNewTag] = useState<Omit<Tag, "tag_id">>({
    key: "",
    value: "",
    budget: 0,
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [tagsPerPage, setTagsPerPage] = useState(5);
  const [searchTerm, setSearchTerm] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchTags();
  }, []);

  const fetchTags = async () => {
    setLoading(true);
    try {
      const response = await axiosInstance.get(`${BACKEND}/tags/tags`);
      const sortedTags = response.data.sort((a: Tag, b: Tag) => a.key.localeCompare(b.key));
      setTags(sortedTags);
    } catch (error) {
      console.error("Error fetching tags:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTag = () => {
    setEditingTag(null);
    setNewTag({ key: "", value: "", budget: 0 });
    setIsDialogOpen(true);
  };

  const handleEditTag = (tag: Tag) => {
    setEditingTag(tag);
    setNewTag({ key: tag.key, value: tag.value, budget: tag.budget });
    setIsDialogOpen(true);
  };

  const handleDeleteTag = async (tagId: number) => {
    try {
      await axiosInstance.delete(`${BACKEND}/tags/tags/${tagId}`);
      setTags((prevTags) => prevTags.filter((tag) => tag.tag_id !== tagId));
      setSelectedTags((prevSelected) => prevSelected.filter(id => id !== tagId));
    } catch (error) {
      console.error("Error deleting tag:", error);
    }
  };

  const handleSaveTag = async () => {
    if (!newTag.key.trim() || !newTag.value.trim()) {
      setErrorMessage("Key and Value are required.");
      setTimeout(() => {
        setErrorMessage(null);
      }, 3000);
      return;
    }
    
    if (/\s/.test(newTag.key) || /\s/.test(newTag.value)) {
      setErrorMessage("Key and Value should not contain spaces.");
      setTimeout(() => {
        setErrorMessage(null);
      }, 3000);
      return;
    }

    const isDuplicate = tags.some(
      (tag) =>
        tag.key === newTag.key &&
        tag.value === newTag.value &&
        tag.tag_id !== (editingTag?.tag_id ?? -1)
    );

    if (isDuplicate) {
      setErrorMessage("A tag with the same key and value already exists.");
      setTimeout(() => {
        setErrorMessage(null);
      }, 3000);
      return;
    }

    try {
      if (editingTag) {
        await axiosInstance.put(`${BACKEND}/tags/tags/${editingTag.tag_id}`, newTag);
      } else {
        await axiosInstance.post(`${BACKEND}/tags/tags`, newTag);
      }
      setIsDialogOpen(false);
      fetchTags();
    } catch (error) {
      console.error("Error saving tag:", error);
    }
  };

  const filteredTags = tags.filter(
    (tag) =>
      tag.key.toLowerCase().includes(searchTerm.toLowerCase()) ||
      tag.value.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const indexOfLastTag = currentPage * tagsPerPage;
  const indexOfFirstTag = indexOfLastTag - tagsPerPage;
  const currentTags = filteredTags.slice(indexOfFirstTag, indexOfLastTag);
  const totalPages = Math.ceil(filteredTags.length / tagsPerPage);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const currentPageTagIds = currentTags.map(tag => tag.tag_id);
      setSelectedTags(currentPageTagIds);
    } else {
      setSelectedTags([]);
    }
  };

  const handleSelectTag = (tagId: number, checked: boolean) => {
    if (checked) {
      setSelectedTags(prev => [...prev, tagId]);
    } else {
      setSelectedTags(prev => prev.filter(id => id !== tagId));
    }
  };

  const isAllCurrentPageSelected = currentTags.length > 0 && 
    currentTags.every(tag => selectedTags.includes(tag.tag_id));

  const handleNewAlertClick = () => {
      setShowNewAlertForm(true);
  };  

  const handleBackClick = () => {
    setShowNewAlertForm(false);
    setIsButtonVisible(true);  // Ensure the button reappears when going back
  };

  const [alertRuleData, setAlertRuleData] = useState<any>(null);

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
        // Optionally, you might want to refresh the project data here
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

  const handleBulkDelete = async () => {
    try {
      await Promise.all(
        selectedTags.map(tagId => 
          axiosInstance.delete(`${BACKEND}/tags/tags/${tagId}`)
        )
      );
      await fetchTags();
      setSelectedTags([]);
    } catch (error) {
      console.error("Error deleting tags:", error);
    }
  };

  const paginate = (pageNumber: number) => {
    setCurrentPage(pageNumber);
    setSelectedTags([]); // Clear selections when changing pages
  };

  const handleTagsPerPageChange = (value: string) => {
    setTagsPerPage(Number(value));
    setCurrentPage(1);
    setSelectedTags([]); // Clear selections when changing items per page
  };

  const handleSaveAlertRule = () => {
    if (alertRuleData) {
      saveAlertRule(alertRuleData);
    }
  };
  
  const renderPaginationItems = () => {
    const items = [];
    const maxVisiblePages = 5;

    if (totalPages <= maxVisiblePages) {
      for (let i = 1; i <= totalPages; i++) {
        items.push(
          <PaginationItem key={i}>
            <PaginationLink
              onClick={() => paginate(i)}
              isActive={currentPage === i}
            >
              {i}
            </PaginationLink>
          </PaginationItem>
        );
      }
    } else {
      const leftBound = Math.max(
        1,
        currentPage - Math.floor(maxVisiblePages / 2)
      );
      const rightBound = Math.min(totalPages, leftBound + maxVisiblePages - 1);

      if (leftBound > 1) {
        items.push(
          <PaginationItem key="start-ellipsis">
            <PaginationEllipsis />
          </PaginationItem>
        );
      }

      for (let i = leftBound; i <= rightBound; i++) {
        items.push(
          <PaginationItem key={i}>
            <PaginationLink
              onClick={() => paginate(i)}
              isActive={currentPage === i}
            >
              {i}
            </PaginationLink>
          </PaginationItem>
        );
      }

      if (rightBound < totalPages) {
        items.push(
          <PaginationItem key="end-ellipsis">
            <PaginationEllipsis />
          </PaginationItem>
        );
      }
    }

    return items;
  };

  return (
    <div className="container mx-auto p-6 md:p-8 lg:p-10 bg-[#F9FEFF] rounded-2xl shadow-lg border border-[#E0E5EF] min-h-screen">
      <h1 className="text-3xl font-extrabold mb-6 text-start text-[#233E7D] tracking-tight">Tag Management</h1>
      <div className="flex justify-between items-center mb-4">
        <div className="flex items-center space-x-2">
          <Button
            onClick={handleCreateTag}
            className="bg-[#233E7D] text-white hover:bg-[#1A2C5B] font-semibold rounded-md shadow px-4 py-2"
          >
            <Plus className="mr-2 h-4 w-4" /> Create New Tag
          </Button>
          {/* {selectedTags.length > 0 && (
            <Button
              onClick={handleBulkDelete}
              variant="destructive"
              className="ml-2"
            >
              <Trash2 className="mr-2 h-4 w-4" /> Delete Selected ({selectedTags.length})
            </Button>
          )} */}
          {selectedTags.length > 0 && (
            <Button
              onClick={handleNewAlertClick}
              className="bg-[#D82026] text-white hover:bg-[#b81a1f] font-semibold rounded-md shadow px-4 py-2"
            >
              <Plus className="h-4 w-4" />
              New Alert Rule
            </Button>
          )}
        </div>
        <div className="relative">
          <Input
            type="text"
            placeholder="Search tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 border border-[#E0E5EF] rounded-md bg-white text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
          />
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#233E7D]" />
        </div>
      </div>
      <Table className="bg-white rounded-lg border border-[#E0E5EF] shadow-sm">
        <TableHeader className="bg-[#F9FEFF]">
          <TableRow>
            <TableHead className="w-[50px]">
              <Checkbox
                checked={isAllCurrentPageSelected}
                onCheckedChange={handleSelectAll}
                aria-label="Select all"
              />
            </TableHead>
            <TableHead className="text-[#233E7D] font-bold px-16 py-6">Key</TableHead>
            <TableHead className="text-[#233E7D] font-bold px-16 py-6">Value</TableHead>
            <TableHead className="text-[#233E7D] font-bold px-16 py-6">Yearly Budget</TableHead>
            <TableHead className="text-[#233E7D] font-bold">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center py-8">
                <div className="h-full w-full flex items-center justify-center">
                  <Loader />
                </div>
              </TableCell>
            </TableRow>
          ) : (
            <AnimatePresence>
              {currentTags.map((tag) => (
                <motion.tr
                  key={tag.tag_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className="hover:bg-[#E0E5EF]/40 transition"
                >
                  <TableCell>
                    <Checkbox
                      checked={selectedTags.includes(tag.tag_id)}
                      onCheckedChange={(checked) => handleSelectTag(tag.tag_id, checked as boolean)}
                      aria-label={`Select tag ${tag.key}`}
                    />
                  </TableCell>
                  <TableCell className="text-[#233E7D] font-medium">{tag.key}</TableCell>
                  <TableCell className="text-[#233E7D]">{tag.value}</TableCell>
                  <TableCell className="text-[#233E7D]">{tag.budget}</TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleEditTag(tag)}
                      className="hover:bg-[#E0E5EF]"
                    >
                      <Edit className="h-4 w-4 text-[#233E7D]" />
                      <span className="sr-only">Edit</span>
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteTag(tag.tag_id)}
                      className="hover:bg-[#D82026]/10"
                    >
                      <Trash2 className="h-4 w-4 text-[#D82026]" />
                      <span className="sr-only">Delete</span>
                    </Button>
                  </TableCell>
                </motion.tr>
              ))}
            </AnimatePresence>
          )}
        </TableBody>
      </Table>
      {/* Render the  component when showAlertRuleForm is true */}
      {showNewAlertForm && (
        <Dialog open={showNewAlertForm} onOpenChange={setShowNewAlertForm}>
          <DialogContent className="w-full max-w-7xl bg-[#F9FEFF] border border-[#E0E5EF] rounded-2xl">
            <DialogHeader>
              <DialogTitle className="text-[#233E7D] font-bold">Create New Alert Rule</DialogTitle>
            </DialogHeader>
            <NewAlertRuleForm
              onBack={handleBackClick}
              onSave={handleSaveAlertRule}
              projectId={projectId}
              selectedTagIds={selectedTags}
            />
          </DialogContent>
        </Dialog>
      )}
      {/* Dialog for creating or editing tags */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="bg-[#F9FEFF] border border-[#E0E5EF] rounded-2xl">
          <DialogHeader>
            <DialogTitle className="text-[#233E7D] font-bold">
              {editingTag ? "Edit Tag" : "Create New Tag"}
            </DialogTitle>
          </DialogHeader>
          {errorMessage && (
            <div className="text-[#D82026] mb-4 font-semibold">
              {errorMessage}
            </div>
          )}
          <div className="grid gap-4 py-4">
            <div className="grid gap-4 py-4">
              <div>
                <Label htmlFor="key" className="text-[#233E7D] font-semibold">Key</Label>
                <Input
                  id="key"
                  type="text"
                  value={newTag.key}
                  onChange={(e) => setNewTag((prev) => ({ ...prev, key: e.target.value }))}
                  placeholder="Enter key"
                  className="border border-[#E0E5EF] rounded-md bg-white text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
                />
              </div>
              <div>
                <Label htmlFor="value" className="text-[#233E7D] font-semibold">Value</Label>
                <Input
                  id="value"
                  type="text"
                  value={newTag.value}
                  onChange={(e) => setNewTag((prev) => ({ ...prev, value: e.target.value }))}
                  placeholder="Enter value"
                  className="border border-[#E0E5EF] rounded-md bg-white text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
                />
              </div>
              <div>
                <Label htmlFor="budget" className="text-[#233E7D] font-semibold">Yearly Budget</Label>
                <Input
                  id="budget"
                  type="number"
                  value={newTag.budget}
                  onChange={(e) => setNewTag((prev) => ({ ...prev, budget: Number(e.target.value) }))}
                  placeholder="Enter yearly budget"
                  className="border border-[#E0E5EF] rounded-md bg-white text-[#233E7D] focus:ring-2 focus:ring-[#233E7D]"
                />
              </div>
            </div>
          </div>
          <Button
            onClick={handleSaveTag}
            className="bg-[#233E7D] hover:bg-[#1A2C5B] text-white font-semibold rounded-md shadow px-4 py-2"
          >
            {editingTag ? "Update Tag" : "Create Tag"}
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  );
};

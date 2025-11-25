"use client";
import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CirclePlus, Pencil, Trash2 } from "lucide-react";
import NewRuleAddContactPoint from "@/components/alertRules/NewRuleAddContactPoint";
import axiosInstance, { BACKEND, fetchAllContactPoints } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import axios from "axios";

// types.ts
export interface ContactPoint {
  id: string;
  name: string;
  integration_type: string;
  url: string;
  notification_template: string;
  project_id: string;
}

type Props = {};

const Page = (props: Props) => {
  const [AddContactPoint, setAddContactPoint] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [contactPoints, setContactPoints] = useState<ContactPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const [editingContactPoint, setEditingContactPoint] = useState<ContactPoint | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);

  const params = useParams();
  const projectId = params.projectName as string;

  const deleteIntegration = async (integrationId: string) => {
    try {
      await axiosInstance.delete(`${BACKEND}/integration/${integrationId}`, {
        headers: {
          accept: "application/json",
        },
      });
      return true;
    } catch (error) {
      console.error("Error deleting integration:", error);
      return false;
    }
  };

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedIntegration, setSelectedIntegration] = useState<ContactPoint | null>(null);

  const handleDeleteClick = (contactPoint: ContactPoint) => {
    setSelectedIntegration(contactPoint);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (selectedIntegration) {
      const success = await deleteIntegration(selectedIntegration.id);
      if (success) {
        setRefreshTrigger((prev) => prev + 1);
        setDeleteDialogOpen(false);
      } else {
        // Handle error (e.g., show an error message)
      }
    }
  };

  useEffect(() => {
    const loadContactPoints = async () => {
      setLoading(true);
      try {
        const data = await fetchAllContactPoints();
        setContactPoints(data);
      } catch (error) {
        setError(error as Error);
      } finally {
        setLoading(false);
      }
    };

    loadContactPoints();
  }, [projectId, refreshTrigger]);

  const handleAddContactPoint = () => {
    setAddContactPoint(true);
  };

  const handleBackClick = () => {
    setAddContactPoint(false);
    setRefreshTrigger((prev) => prev + 1);
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const filteredContactPoints = contactPoints.filter(
    (contactPoint) =>
      contactPoint.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      contactPoint.integration_type
        .toLowerCase()
        .includes(searchQuery.toLowerCase())
  );

  // Handle editing changes
  const handleEditChange = (field: keyof ContactPoint, value: string) => {
    if (editingContactPoint) {
      setEditingContactPoint({
        ...editingContactPoint,
        [field]: value,
      });
    }
  };

  // Handle confirm edit
  const handleEditConfirm = async () => {
    if (editingContactPoint?.id) {
      const { id, ...updatedContactPoint } = editingContactPoint;  // Destructure to remove 'id' from the update payload
  
      // Ensure notification_template is included as an empty object if not provided
      const payload = {
        ...updatedContactPoint,
        notification_template: updatedContactPoint.notification_template || {}, // Set empty object if not present
      };
  
      try {
        await axiosInstance.put(
          `${BACKEND}/integration/${editingContactPoint.id}`,
          payload // Send the updated contact point without the 'id' field
        );
        setRefreshTrigger((prev) => prev + 1); // Refresh the data
        setEditDialogOpen(false); // Close the dialog
      } catch (error) {
        console.error("Error updating contact point:", error);
        // Handle error (e.g., show an error message)
      }
    }
  };   

  const handleEditClick = (contactPoint: ContactPoint) => {
    setEditingContactPoint(contactPoint);
    setEditDialogOpen(true);
  };

  return (
    <div className="bg-[#F9FEFF] min-h-screen w-full p-6">
      <div className="bg-white h-full shadow-md border border-[#E0E5EF] rounded-md p-6 w-full overflow-y-auto">
        <div className="text-3xl font-bold text-[#233E7D] mb-6">Contact Points</div>
        {!AddContactPoint ? (
          <div>
            <div className="mb-6 flex justify-between items-center">
              <Input
                placeholder="Search by name or type"
                className="flex-1 mr-4 border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base"
                value={searchQuery}
                onChange={handleSearchChange}
              />
              <Button
                className="gap-2 bg-[#D82026] hover:bg-[#b81a1f] text-white text-sm font-medium rounded-md h-10 px-4 py-2"
                onClick={handleAddContactPoint}
              >
                <CirclePlus size={16} />
                Add Contact Point
              </Button>
            </div>

            {loading ? (
              <div className="mt-4"></div>
            ) : error ? (
              <div className="mt-4 text-[#D82026] text-base">Error: {error.message}</div>
            ) : (
              <div className="mt-6 space-y-4">
                {filteredContactPoints.map((contactPoint) => (
                  <div
                    key={contactPoint.id}
                    className="w-full border border-[#E0E5EF] rounded-md shadow-md hover:shadow-lg transition-all bg-white hover:border-[#233E7D]"
                  >
                    <div className="w-full border-b border-[#E0E5EF] p-4 flex justify-between items-center bg-[#F9FEFF]">
                      <div className="text-lg font-semibold text-[#233E7D]">
                        {contactPoint.name}
                      </div>
                      <div className="flex items-center gap-6">
                        <button
                          onClick={() => handleEditClick(contactPoint)}
                          className="flex gap-2 items-center text-[#233E7D] hover:text-[#D82026] text-sm font-medium"
                        >
                          <Pencil size={16} />
                          Edit
                        </button>
                        <button
                          onClick={() => handleDeleteClick(contactPoint)}
                          className="text-[#D82026] hover:text-[#b81a1f] text-sm font-medium"
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </div>
                    <div className="p-4 text-base text-[#6B7280]">
                      Integration Type:{" "}
                      <span className="font-medium text-[#233E7D]">{contactPoint.integration_type}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <NewRuleAddContactPoint
            onBack={handleBackClick}
            onSave={handleBackClick}
          />
        )}

        {/* Edit Dialog */}
        <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
          <DialogContent className="sm:max-w-[425px] bg-white rounded-md">
            <DialogHeader>
              <DialogTitle className="text-[#233E7D] text-2xl font-bold">Edit Contact Point</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="flex flex-col">
                <label htmlFor="name" className="text-sm font-medium text-[#233E7D]">
                  Name
                </label>
                <Input
                  id="name"
                  value={editingContactPoint?.name || ""}
                  onChange={(e) =>
                    handleEditChange("name", e.target.value)
                  }
                  className="border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base"
                />
              </div>

              <div className="flex flex-col">
                <label htmlFor="integration_type" className="text-sm font-medium text-[#233E7D]">
                  Integration Type
                </label>
                <Input
                  id="integration_type"
                  value={editingContactPoint?.integration_type || ""}
                  onChange={(e) =>
                    handleEditChange("integration_type", e.target.value)
                  }
                  className="border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base"
                />
              </div>

              <div className="flex flex-col">
                <label htmlFor="url" className="text-sm font-medium text-[#233E7D]">
                  URL
                </label>
                <Input
                  id="url"
                  value={editingContactPoint?.url || ""}
                  onChange={(e) =>
                    handleEditChange("url", e.target.value)
                  }
                  className="border-[#C8C8C8] rounded-md px-3 py-2 placeholder:text-gray-400 focus:ring-2 focus:ring-[#233E7D] text-base"
                />
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setEditDialogOpen(false)}
                className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md"
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleEditConfirm}
                className="bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md"
              >
                Save Changes
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Dialog */}
        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent className="sm:max-w-[425px] bg-white rounded-md">
            <DialogHeader>
              <DialogTitle className="text-[#D82026] text-2xl font-bold">Confirm Deletion</DialogTitle>
              <DialogDescription className="text-base text-[#6B7280]">
                Are you sure you want to delete this integration? This action cannot be undone.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setDeleteDialogOpen(false)}
                className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10 text-sm font-medium rounded-md"
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDeleteConfirm}
                className="bg-[#D82026] text-white hover:bg-[#b81a1f] text-sm font-medium rounded-md"
              >
                Delete
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
};

export default Page;

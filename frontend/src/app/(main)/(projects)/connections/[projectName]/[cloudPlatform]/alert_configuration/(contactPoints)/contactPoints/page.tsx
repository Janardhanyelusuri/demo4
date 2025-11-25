"use client";
import React, { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CirclePlus, Pencil, Trash2 } from "lucide-react";
import RuleAddContactPoint from "@/components/alertRules/RuleAddContactPoint";
import axiosInstance, { BACKEND, fetchContactPoints } from "@/lib/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

// types.ts
export interface ContactPoint {
  id: string;
  name: string;
  integration_type: string;
  url: string;
  notification_template: string; // Will pass as an empty string but won't show in UI
  project_id: string;
}

type Props = {};

const Page = (props: Props) => {
  const [AddContactPoint, setAddContactPoint] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [Points, setContactPoints] = useState<ContactPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [editContactPoint, setEditContactPoint] = useState<ContactPoint | null>(null); // For editing

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
  const [selectedIntegration, setSelectedIntegration] =
    useState<ContactPoint | null>(null);

  const handleDeleteClick = (Point: ContactPoint) => {
    setSelectedIntegration(Point);
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

  const handleEditClick = (Point: ContactPoint) => {
    setEditContactPoint(Point); // Set selected  point to edit
  };

  const handleEditConfirm = async () => {
    if (editContactPoint) {
      // Exclude the `id` from the data being sent to the backend
      const { id, ...updatedData } = editContactPoint;
  
      const updatedContactPoint = {
        ...updatedData,
        notification_template: {}, // Ensure it's passed as an empty JSON object
      };
  
      try {
        // Make the API call to update the  point (without the `id` field)
        await axiosInstance.put(`${BACKEND}/integration/${editContactPoint.id}`, updatedContactPoint);
        setEditContactPoint(null); // Close the dialog
        setRefreshTrigger((prev) => prev + 1); // Trigger refresh
      } catch (error) {
        console.error("Error updating  point:", error);
      }
    }
  };  

  useEffect(() => {
    const loadContactPoints = async () => {
      setLoading(true);
      try {
        const data = await fetchContactPoints(projectId);
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

  const filteredContactPoints = Points.filter(
    (Point) =>
      Point.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      Point.integration_type
        .toLowerCase()
        .includes(searchQuery.toLowerCase())
  );

  return (
    <div className="bg-white h-full shadow-md border-2 rounded-md p-4 w-full overflow-y-auto">
      <div className="text-2xl mt-4 text-[#233E7D] font-semibold">Contact Points</div>
      {!AddContactPoint && !editContactPoint ? (
        <div>
          <div className="mt-4 flex justify-between items-center">
            <Input
              placeholder="Search by name or type"
              className="flex-1 mr-4"
              value={searchQuery}
              onChange={handleSearchChange}
            />
            <Button
              variant="default"
              className="text-xs flex gap-2 items-center bg-[#D82026] hover:bg-[#b81a1f] text-white"
              onClick={handleAddContactPoint}
            >
              <CirclePlus size={16} />
              Add Contact point
            </Button>
          </div>

          {loading ? (
            <div className="mt-4">Loading...</div>
          ) : error ? (
            <div className="mt-4 text-[#D82026]">Error: {error.message}</div>
          ) : (
            filteredContactPoints.map((Point) => (
              <div
                key={Point.id}
                className="w-full border border-[#C8C8C8] rounded-sm mt-4"
              >
                <div className="w-full border-b p-4 flex justify-between items-center bg-gray-100 hover:bg-[#233E7D]/10 transition-colors">
                  <div className="text-sm font-semibold text-[#233E7D]">
                    {Point.name}
                  </div>
                  <div className="flex items-center gap-6">
                    <button
                      onClick={() => handleEditClick(Point)}
                      className="flex gap-2 items-center text-[#233E7D] hover:text-[#D82026] transition-colors"
                      title="Edit Contact Point"
                    >
                      <Pencil size={16} />
                      Edit
                    </button>
                    <button
                      onClick={() => handleDeleteClick(Point)}
                      className="text-[#D82026] hover:text-[#b81a1f] transition-colors"
                      title="Delete Contact Point"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                </div>
                <div className="p-4 text-sm text-gray-600">
                  Integration Type:{" "}
                  <span className="font-medium text-[#233E7D]">{Point.integration_type}</span>
                </div>
              </div>
            ))
          )}
        </div>
      ) : AddContactPoint ? (
        <RuleAddContactPoint
          onBack={handleBackClick}
          onSave={handleBackClick}
          projectId={projectId}
        />
      ) : (
<Dialog open={!!editContactPoint} onOpenChange={() => setEditContactPoint(null)}>
  <DialogContent className="sm:max-w-[425px]">
    <DialogHeader>
      <DialogTitle className="text-[#233E7D]">Edit Contact Point</DialogTitle>
    </DialogHeader>
    <div className="space-y-4 p-4">
      {/* Name input */}
      <div>
        <label htmlFor="name" className="block text-sm font-medium text-[#233E7D]">
          Name
        </label>
        <Input
          id="name"
          value={editContactPoint?.name || ""}
          onChange={(e) => setEditContactPoint({ ...editContactPoint!, name: e.target.value })}
        />
      </div>

      {/* Integration Type input */}
      <div>
        <label htmlFor="integration_type" className="block text-sm font-medium text-[#233E7D]">
          Integration Type
        </label>
        <Input
          id="integration_type"
          value={editContactPoint?.integration_type || ""}
          onChange={(e) => setEditContactPoint({ ...editContactPoint!, integration_type: e.target.value })}
        />
      </div>

      {/* URL input */}
      <div>
        <label htmlFor="url" className="block text-sm font-medium text-[#233E7D]">
          URL
        </label>
        <Input
          id="url"
          value={editContactPoint?.url || ""}
          onChange={(e) => setEditContactPoint({ ...editContactPoint!, url: e.target.value })}
        />
      </div>
    </div>
    <DialogFooter>
      <Button
        variant="outline"
        onClick={() => setEditContactPoint(null)}
        className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
      >
        Cancel
      </Button>
      <Button
        variant="default"
        onClick={handleEditConfirm}
        className="bg-[#D82026] hover:bg-[#b81a1f] text-white"
      >
        Save Changes
      </Button>
    </DialogFooter>
  </DialogContent>
</Dialog>
      )}

      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle className="text-[#D82026]">Confirm Deletion</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this integration? This action
              cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteDialogOpen(false)}
              className="border-[#233E7D] text-[#233E7D] hover:bg-[#233E7D]/10"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteConfirm}
              className="bg-[#D82026] hover:bg-[#b81a1f] text-white"
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Page;

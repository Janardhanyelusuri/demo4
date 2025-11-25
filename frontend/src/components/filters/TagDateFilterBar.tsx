"use client";
import React, { useEffect, useState, useRef } from "react";
import { ChevronDownIcon } from "@heroicons/react/20/solid";
import axiosInstance, { BACKEND } from "@/lib/api";

interface Tag {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface TagDateFilterBarProps {
  onTagSelect?: (tagId: number | undefined) => void;
  onDateRangeSelect?: (range: string | undefined) => void;
}

const dateOptions = [
  { label: "Today", value: "today" },
  { label: "Yesterday", value: "yesterday" },
  { label: "Last 7 days", value: "last_7_days" },
  { label: "Last 30 days", value: "last_30_days" },
  { label: "Last 90 days", value: "last_90_days" },
  { label: "This week", value: "this_week" },
  { label: "Last week", value: "last_week" },
  { label: "This month", value: "this_month" },
  { label: "Last month", value: "last_month" },
  { label: "This year", value: "this_year" },
  { label: "Last year", value: "last_year" },
];

const DropdownWrapper = ({ children }: { children: React.ReactNode }) => (
  <div className="relative w-40">
    {children}
    <ChevronDownIcon className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-500 pointer-events-none" />
  </div>
);

const TagDateFilterBar: React.FC<TagDateFilterBarProps> = ({ onTagSelect, onDateRangeSelect }) => {
  // Tag filter state
  const [tags, setTags] = useState<Tag[]>([]);
  const [tempKey, setTempKey] = useState<string | null>(null);
  const [tempValue, setTempValue] = useState<string | null>(null);
  const [selectedTagId, setSelectedTagId] = useState<number | undefined>(undefined);

  // Date filter state
  const [tempDate, setTempDate] = useState<string | null>(null);
  const [appliedDate, setAppliedDate] = useState<string | undefined>(undefined);

  const hasFetchedTags = useRef(false);
  useEffect(() => {
    if (hasFetchedTags.current) return;
    hasFetchedTags.current = true;
    const fetchTags = async () => {
      try {
        const response = await axiosInstance.get(`${BACKEND}/tags/tags`);
        setTags(response.data);
      } catch (error) {
        console.error("Error fetching tags:", error);
      }
    };
    fetchTags();
  }, []);

  const uniqueKeys = Array.from(new Set(tags.map((tag) => tag.key)));
  const valuesForSelectedKey = tempKey
    ? tags.filter((tag) => tag.key === tempKey).map((tag) => tag.value)
    : [];

  const handleKeyChange = (value: string) => {
    setTempKey(value);
    setTempValue(null);
  };

  const handleApply = () => {
    // Tag logic
    if (tempKey && tempValue) {
      const selectedTag = tags.find(tag => tag.key === tempKey && tag.value === tempValue);
      if (selectedTag) {
        setSelectedTagId(selectedTag.tag_id);
        if (onTagSelect) onTagSelect(selectedTag.tag_id);
      }
    } else {
      setSelectedTagId(undefined);
      if (onTagSelect) onTagSelect(undefined);
    }
    // Date logic
    setAppliedDate(tempDate || undefined);
    if (onDateRangeSelect) onDateRangeSelect(tempDate || undefined);
  };

  // Clear all filters handler
  const handleClearAll = () => {
    setTempKey(null);
    setTempValue(null);
    setSelectedTagId(undefined);
    setTempDate(null);
    setAppliedDate(undefined);
    if (onTagSelect) onTagSelect(undefined);
    if (onDateRangeSelect) onDateRangeSelect(undefined);
    setTimeout(() => {
      handleApply();
    }, 0);
  };

return (
    <div className="flex items-center px-6 py-3 bg-[#F9FEFF] rounded-xl shadow border border-[#233E7D]/20 max-w-full overflow-x-auto">
      <div className="flex space-x-6 items-center flex-grow">
        {/* Tag Key */}
        <DropdownWrapper>
          <select
            aria-label="Tag Key"
            className="appearance-none w-full bg-[#EAF1FB] text-[#233E7D] text-xs font-semibold px-3 py-2 pr-6 rounded-lg shadow-sm border border-[#B6C6E3] focus:outline-none focus:ring-2 focus:ring-[#233E7D]/40 focus:border-[#233E7D] hover:bg-[#D6E4F7] cursor-pointer transition-colors"
            value={tempKey || ""}
            onChange={(e) => handleKeyChange(e.target.value)}
          >
            <option value="" disabled>
              Select Tag Key
            </option>
            {uniqueKeys.map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </DropdownWrapper>

        {/* Tag Value */}
        <DropdownWrapper>
          <select
            aria-label="Tag Value"
            title="Tag Value"
            className={`appearance-none w-full text-[#233E7D] text-xs font-semibold px-3 py-2 pr-6 rounded-lg shadow-sm border transition duration-150 ${
              tempKey
                ? "bg-[#EAF1FB] border-[#B6C6E3] focus:outline-none focus:ring-2 focus:ring-[#233E7D]/40 focus:border-[#233E7D] hover:bg-[#D6E4F7] cursor-pointer"
                : "bg-gray-50 border-gray-200 cursor-not-allowed"
            }`}
            value={tempValue || ""}
            onChange={(e) => setTempValue(e.target.value)}
            disabled={!tempKey}
          >
            <option value="" disabled>
              Select Tag Value
            </option>
            {valuesForSelectedKey.map((val, idx) => (
              <option key={idx} value={val}>
                {val}
              </option>
            ))}
          </select>
        </DropdownWrapper>

        {/* Date Range */}
        <DropdownWrapper>
          <select
            aria-label="Date Range"
            className="appearance-none w-full bg-[#EAF1FB] text-[#233E7D] text-xs font-semibold px-3 py-2 pr-6 rounded-lg shadow-sm border border-[#B6C6E3] focus:outline-none focus:ring-2 focus:ring-[#233E7D]/40 focus:border-[#233E7D] hover:bg-[#D6E4F7] cursor-pointer transition-colors"
            value={tempDate || ""}
            onChange={(e) => setTempDate(e.target.value)}
          >
            <option value="" disabled>
              Select Date Range
            </option>
            {dateOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </DropdownWrapper>

        {/* Apply Button */}
        <button
          onClick={handleApply}
          disabled={!(tempKey && tempValue) && !tempDate}
          className={`px-4 py-2 rounded-md text-xs font-bold transition-colors duration-150 shadow-sm border border-[#233E7D] ${
            ((tempKey && tempValue) || tempDate)
              ? "bg-[#233E7D] text-white hover:bg-[#1a2d5c]"
              : "bg-[#B6C6E3] text-white cursor-not-allowed border-[#B6C6E3]"
          }`}
        >
          Apply
        </button>
      </div>
      {/* Clear All Button */}
      <button
        onClick={handleClearAll}
        className="ml-4 px-4 py-2 rounded-md text-xs font-bold bg-white text-[#233E7D] hover:bg-[#EAF1FB] transition-colors duration-150 border border-[#233E7D] shadow-sm"
        type="button"
      >
        Clear All
      </button>
    </div>
  );
};

export default TagDateFilterBar;

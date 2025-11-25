import React from "react";

// Tag interface specific to TagFilterSelect
interface Tags {
  tag_id: number;
  key: string;
  value: string;
  budget: number;
}

interface TagFilterSelectProps {
  tags: Tags[];
  selectedTagId?: number;
  onChange: (tagId: number | undefined) => void;
}

const TagFilterSelect: React.FC<TagFilterSelectProps> = ({
  tags,
  selectedTagId,
  onChange,
}) => {
  const handleTagChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    
    // If value is empty string, pass undefined, otherwise convert to number
    const tagId = value === "" ? undefined : parseInt(value, 10);
    
    // Log the selected tag ID for debugging
    console.log("Selected tag_id:", tagId);

    onChange(tagId);
  };

  return (
    <div className="mb-4">
      <label
        htmlFor="tag-select"
        className="block text-sm font-semibold text-[#233E7D] mb-1"
      >
        Filter by Tag
      </label>
      <select
        id="tag-select"
        value={selectedTagId !== undefined ? selectedTagId.toString() : ""}
        onChange={handleTagChange}
        className="border border-[#C8C8C8] rounded-md px-3 py-2 w-full md:w-64 text-[#233E7D] bg-[#F9FEFF] focus:ring-2 focus:ring-[#233E7D] font-medium transition-colors duration-150 hover:border-[#233E7D]"
      >
        <option value="" className="text-[#6B7280]">All Tags</option>
        {tags.map((tag) => (
          <option key={tag.tag_id} value={tag.tag_id.toString()} className="text-[#233E7D]">
            {tag.key}: {tag.value}
          </option>
        ))}
      </select>
    </div>
  );
};

export default TagFilterSelect;
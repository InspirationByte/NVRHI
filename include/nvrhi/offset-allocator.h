// (C) Sebastian Aaltonen 2023
// License: MIT License (https://github.com/sebbbi/OffsetAllocator/blob/main/LICENSE)

#pragma once
#include <cstdint>
#include <climits>

namespace nvrhi
{
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

class OffsetAllocator
{
public:
	using NodeIndex = uint32;

	static constexpr uint32 NUM_TOP_BINS = 32;
	static constexpr uint32 BINS_PER_LEAF = 8;
	static constexpr uint32 TOP_BINS_INDEX_SHIFT = 3;
	static constexpr uint32 LEAF_BINS_INDEX_MASK = 0x7;
	static constexpr uint32 NUM_LEAF_BINS = NUM_TOP_BINS * BINS_PER_LEAF;

	struct Alloc
	{
		static constexpr uint32 NO_SPACE = 0xffffffff;

		uint32		offset = NO_SPACE;
		NodeIndex	metadata = NO_SPACE; // internal: node index
	};

	struct StorageReport
	{
		uint32		totalFreeSpace = 0;
		uint32		largestFreeRegion = 0;
	};

	struct StorageReportFull
	{
		struct Region
		{
			uint32	size = 0;
			uint32	count = 0;
		};

		Region		freeRegions[NUM_LEAF_BINS];
	};

	~OffsetAllocator();
	OffsetAllocator(uint32 maxAllocs = USHRT_MAX);
	OffsetAllocator(OffsetAllocator&& other);

	void				reset(uint32 newSize);

	Alloc				allocate(uint32 size);
	void				free(Alloc allocation) { free(allocation.metadata); }
	void				free(NodeIndex nodeIndex);

	uint32				allocationSize(Alloc allocation) const;
	StorageReport		storageReport() const;
	StorageReportFull	storageReportFull() const;

private:
	uint32				insertNodeIntoBin(uint32 size, uint32 dataOffset);
	void				removeNodeFromBin(NodeIndex nodeIndex);

	struct Node;

	uint32				m_size{ 0 };
	uint32				m_maxAllocs{ 0 };
	uint32				m_freeStorage{ 0 };

	uint32				m_usedBinsTop;
	uint8				m_usedBins[NUM_TOP_BINS];
	NodeIndex			m_binIndices[NUM_LEAF_BINS];

	Node*				m_nodes{ nullptr };
	NodeIndex*			m_freeNodes{ nullptr };
	uint32				m_freeOffset{ 0 };
};

inline OffsetAllocator::Alloc operator+(const OffsetAllocator::Alloc& a, uint32 b)			{ return { a.offset + b, a.metadata }; }
inline bool operator==(const OffsetAllocator::Alloc& a, const OffsetAllocator::Alloc& b)	{ return a.metadata == b.metadata; }
inline bool operator!=(const OffsetAllocator::Alloc& a, const OffsetAllocator::Alloc& b)	{ return !(a == b); }

}
// (C) Sebastian Aaltonen 2023
// License: MIT License (https://github.com/sebbbi/OffsetAllocator/blob/main/LICENSE)

#include <nvrhi/offset-allocator.h>

#ifdef DEBUG
#include <assert.h>
#define ASSERT(x) assert(x)
//#define DEBUG_VERBOSE
#else
#define ASSERT(x)
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#ifdef DEBUG_VERBOSE
#include <stdio.h>
#endif

#include <cstring>

namespace nvrhi
{
inline uint32 lzcnt_nonzero(uint32 v)
{
#ifdef _MSC_VER
    unsigned long retVal;
    _BitScanReverse(&retVal, v);
    return 31 - retVal;
#else
    return __builtin_clz(v);
#endif
}

inline uint32 tzcnt_nonzero(uint32 v)
{
#ifdef _MSC_VER
    unsigned long retVal;
    _BitScanForward(&retVal, v);
    return retVal;
#else
    return __builtin_ctz(v);
#endif
}

static constexpr uint32 SMALL_FLT_MANTISSA_BITS = 3;
static constexpr uint32 SMALL_FLT_MANTISSA_VALUE = 1 << SMALL_FLT_MANTISSA_BITS;
static constexpr uint32 SMALL_FLT_MANTISSA_MASK = SMALL_FLT_MANTISSA_VALUE - 1;

// Bin sizes follow floating point (exponent + mantissa) distribution (piecewise linear log approx)
// This ensures that for each size class, the average overhead percentage stays the same
static uint32 uintToSmallFloatRoundUp(uint32 size)
{
	uint32 exp = 0;
	uint32 mantissa = 0;

	if (size < SMALL_FLT_MANTISSA_VALUE)
	{
		// Denorm: 0..(MANTISSA_VALUE-1)
		mantissa = size;
	}
	else
	{
		// Normalized: Hidden high bit always 1. Not stored. Just like float.
		const uint32 leadingZeros = lzcnt_nonzero(size);
		const uint32 highestSetBit = 31 - leadingZeros;

		const uint32 mantissaStartBit = highestSetBit - SMALL_FLT_MANTISSA_BITS;
		exp = mantissaStartBit + 1;
		mantissa = (size >> mantissaStartBit) & SMALL_FLT_MANTISSA_MASK;

		const uint32 lowBitsMask = (1 << mantissaStartBit) - 1;

		// Round up!
		if ((size & lowBitsMask) != 0)
			mantissa++;
	}

	return (exp << SMALL_FLT_MANTISSA_BITS) + mantissa; // + allows mantissa->exp overflow for round up
}

static uint32 uintToSmallFloatRoundDown(uint32 size)
{
	uint32 exp = 0;
	uint32 mantissa = 0;

	if (size < SMALL_FLT_MANTISSA_VALUE)
	{
		mantissa = size; // Denorm: 0..(MANTISSA_VALUE-1)
	}
	else
	{
		// Normalized: Hidden high bit always 1. Not stored. Just like float.
		const uint32 leadingZeros = lzcnt_nonzero(size);
		const uint32 highestSetBit = 31 - leadingZeros;

		const uint32 mantissaStartBit = highestSetBit - SMALL_FLT_MANTISSA_BITS;
		exp = mantissaStartBit + 1;
		mantissa = (size >> mantissaStartBit) & SMALL_FLT_MANTISSA_MASK;
	}

	return (exp << SMALL_FLT_MANTISSA_BITS) | mantissa;
}

static uint32 smallFloatToUint(uint32 floatValue)
{
	const uint32 exponent = floatValue >> SMALL_FLT_MANTISSA_BITS;
	const uint32 mantissa = floatValue & SMALL_FLT_MANTISSA_MASK;
	if (exponent == 0)
		return mantissa; // Denorms
	else
		return (mantissa | SMALL_FLT_MANTISSA_VALUE) << (exponent - 1);
}

// Utility functions
static uint32 findLowestSetBitAfter(uint32 bitMask, uint32 startBitIndex)
{
	const uint32 maskBeforeStartIndex = (1 << startBitIndex) - 1;
	const uint32 maskAfterStartIndex = ~maskBeforeStartIndex;
	const uint32 bitsAfter = bitMask & maskAfterStartIndex;
	if (bitsAfter == 0)
		return OffsetAllocator::Alloc::NO_SPACE;

	return tzcnt_nonzero(bitsAfter);
}

// Allocator...
OffsetAllocator::OffsetAllocator(uint32 maxAllocs)
	: m_size(0)
	, m_maxAllocs(maxAllocs)
{
	if constexpr (sizeof(NodeIndex) == 2)
	{
		ASSERT(maxAllocs <= USHRT_MAX) //  maxAllocs is limited by USE_16_BIT_NODE_INDICES
	}
}

OffsetAllocator::OffsetAllocator(OffsetAllocator&& other) :
	m_size(other.m_size),
	m_maxAllocs(other.m_maxAllocs),
	m_freeStorage(other.m_freeStorage),
	m_usedBinsTop(other.m_usedBinsTop),
	m_nodes(other.m_nodes),
	m_freeNodes(other.m_freeNodes),
	m_freeOffset(other.m_freeOffset)
{
	memcpy(m_usedBins, other.m_usedBins, sizeof(uint8) * NUM_TOP_BINS);
	memcpy(m_binIndices, other.m_binIndices, sizeof(NodeIndex) * NUM_LEAF_BINS);

	other.m_nodes = nullptr;
	other.m_freeNodes = nullptr;
	other.m_freeOffset = 0;
	other.m_maxAllocs = 0;
	other.m_usedBinsTop = 0;
}

struct OffsetAllocator::Node
{
	static constexpr NodeIndex UNUSED = 0xffffffff;

	Node() = default;
	Node(uint32 dataOffset, uint32 dataSize, NodeIndex binListNext)
		: dataOffset(dataOffset)
		, dataSize(dataSize)
		, binListNext(binListNext)
	{
	}

	uint32		dataOffset = 0;
	uint32		dataSize = 0;
	NodeIndex	binListPrev = UNUSED;
	NodeIndex	binListNext = UNUSED;
	NodeIndex	neighborPrev = UNUSED;
	NodeIndex	neighborNext = UNUSED;
	bool		used = false; // TODO: Merge as bit flag
};

void OffsetAllocator::reset(uint32 newSize)
{
	if (m_size == newSize)
		return;

	m_size = newSize;
	m_freeStorage = 0;
	m_usedBinsTop = 0;
	m_freeOffset = m_maxAllocs;

	for (uint32 i = 0; i < NUM_TOP_BINS; i++)
		m_usedBins[i] = 0;

	for (uint32 i = 0; i < NUM_LEAF_BINS; i++)
		m_binIndices[i] = Node::UNUSED;

	delete [] m_nodes;
	delete [] m_freeNodes;

	m_nodes = new Node[m_maxAllocs + 1];
	m_freeNodes = new NodeIndex[m_maxAllocs + 1];

	// Freelist is a stack. Nodes in inverse order so that [0] pops first.
	for (uint32 i = 0; i < m_maxAllocs + 1; i++)
		m_freeNodes[i] = m_maxAllocs - i;

	// Start state: Whole storage as one big node
	// Algorithm will split remainders and push them back as smaller nodes
	insertNodeIntoBin(m_size, 0);
}

OffsetAllocator::~OffsetAllocator()
{
	delete [] m_nodes;
	delete [] m_freeNodes;
}

OffsetAllocator::Alloc OffsetAllocator::allocate(uint32 size)
{
	// Out of allocations?
	if (m_freeOffset == Alloc::NO_SPACE)
		return {};

	// Round up to bin index to ensure that alloc >= bin
	// Gives us min bin index that fits the size
	const uint32 minBinIndex = uintToSmallFloatRoundUp(size);
	const uint32 minTopBinIndex = minBinIndex >> TOP_BINS_INDEX_SHIFT;
	const uint32 minLeafBinIndex = minBinIndex & LEAF_BINS_INDEX_MASK;

	uint32 topBinIndex = minTopBinIndex;
	uint32 leafBinIndex = Alloc::NO_SPACE;

	// If top bin exists, scan its leaf bin. This can fail (NO_SPACE).
	if (m_usedBinsTop & (1 << topBinIndex))
	{
		leafBinIndex = findLowestSetBitAfter(m_usedBins[topBinIndex], minLeafBinIndex);
	}

	// If we didn't find space in top bin, we search top bin from +1
	if (leafBinIndex == Alloc::NO_SPACE)
	{
		topBinIndex = findLowestSetBitAfter(m_usedBinsTop, minTopBinIndex + 1);

		// Out of space?
		if (topBinIndex == Alloc::NO_SPACE)
			return {};

		// All leaf bins here fit the alloc, since the top bin was rounded up. Start leaf search from bit 0.
		// NOTE: This search can't fail since at least one leaf bit was set because the top bit was set.
		leafBinIndex = tzcnt_nonzero(m_usedBins[topBinIndex]);
	}

	const uint32 binIndex = (topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex;

	// Pop the top node of the bin. Bin top = node.next.
	const uint32 nodeIndex = m_binIndices[binIndex];
	Node& node = m_nodes[nodeIndex];
	const uint32 nodeTotalSize = node.dataSize;
	node.dataSize = size;
	node.used = true;
	m_binIndices[binIndex] = node.binListNext;

	if (node.binListNext != Node::UNUSED)
		m_nodes[node.binListNext].binListPrev = Node::UNUSED;
	m_freeStorage -= nodeTotalSize;

#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Free storage: %u (-%u) (allocate)\n", m_freeStorage, nodeTotalSize);
#endif

	// Bin empty?
	if (m_binIndices[binIndex] == Node::UNUSED)
	{
		// Remove a leaf bin mask bit
		m_usedBins[topBinIndex] &= ~(1 << leafBinIndex);

		// All leaf bins empty?
		if (m_usedBins[topBinIndex] == 0)
		{
			// Remove a top bin mask bit
			m_usedBinsTop &= ~(1 << topBinIndex);
		}
	}

	// Push back reminder N elements to a lower bin
	const uint32 reminderSize = nodeTotalSize - size;
	if (reminderSize > 0)
	{
		const uint32 newNodeIndex = insertNodeIntoBin(reminderSize, node.dataOffset + size);

		// Link nodes next to each other so that we can merge them later if both are free
		// And update the old next neighbor to point to the new node (in middle)
		if (node.neighborNext != Node::UNUSED)
			m_nodes[node.neighborNext].neighborPrev = newNodeIndex;
		m_nodes[newNodeIndex].neighborPrev = nodeIndex;
		m_nodes[newNodeIndex].neighborNext = node.neighborNext;
		node.neighborNext = newNodeIndex;
	}

	return Alloc{ node.dataOffset, nodeIndex };
}

void OffsetAllocator::free(uint32 nodeIndex)
{
	ASSERT(nodeIndex != Alloc::NO_SPACE) // Allocation is not valid

	if (nodeIndex == Alloc::NO_SPACE || !m_nodes)
		return;

	Node& node = m_nodes[nodeIndex];

	ASSERT(node.used == true) //  Double free on node

	// Merge with neighbors...
	uint32 offset = node.dataOffset;
	uint32 size = node.dataSize;

	if (node.neighborPrev != Node::UNUSED && m_nodes[node.neighborPrev].used == false)
	{
		// Previous (contiguous) free node: Change offset to previous node offset. Sum sizes
		const Node& prevNode = m_nodes[node.neighborPrev];
		offset = prevNode.dataOffset;
		size += prevNode.dataSize;

		// Remove node from the bin linked list and put it in the freelist
		removeNodeFromBin(node.neighborPrev);

		ASSERT(prevNode.neighborNext == nodeIndex);
		node.neighborPrev = prevNode.neighborPrev;
	}

	if (node.neighborNext != Node::UNUSED && m_nodes[node.neighborNext].used == false)
	{
		// Next (contiguous) free node: Offset remains the same. Sum sizes.
		const Node& nextNode = m_nodes[node.neighborNext];
		size += nextNode.dataSize;

		// Remove node from the bin linked list and put it in the freelist
		removeNodeFromBin(node.neighborNext);

		ASSERT(nextNode.neighborPrev == nodeIndex);
		node.neighborNext = nextNode.neighborNext;
	}

	const uint32 neighborNext = node.neighborNext;
	const uint32 neighborPrev = node.neighborPrev;

	// Insert the removed node to freelist
#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Putting node %u into freelist[%u] (free)\n", nodeIndex, m_freeOffset + 1);
#endif
	m_freeNodes[++m_freeOffset] = nodeIndex;

	// Insert the (combined) free node to bin
	const uint32 combinedNodeIndex = insertNodeIntoBin(size, offset);

	// Connect neighbors with the new combined node
	if (neighborNext != Node::UNUSED)
	{
		m_nodes[combinedNodeIndex].neighborNext = neighborNext;
		m_nodes[neighborNext].neighborPrev = combinedNodeIndex;
	}
	if (neighborPrev != Node::UNUSED)
	{
		m_nodes[combinedNodeIndex].neighborPrev = neighborPrev;
		m_nodes[neighborPrev].neighborNext = combinedNodeIndex;
	}
}

uint32 OffsetAllocator::insertNodeIntoBin(uint32 size, uint32 dataOffset)
{
	// Round down to bin index to ensure that bin >= alloc
	const uint32 binIndex = uintToSmallFloatRoundDown(size);

	const uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
	const uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

	// Bin was empty before?
	if (m_binIndices[binIndex] == Node::UNUSED)
	{
		// Set bin mask bits
		m_usedBins[topBinIndex] |= 1 << leafBinIndex;
		m_usedBinsTop |= 1 << topBinIndex;
	}

	// Take a freelist node and insert on top of the bin linked list (next = old top)
	const uint32 topNodeIndex = m_binIndices[binIndex];
	const uint32 nodeIndex = m_freeNodes[m_freeOffset--];
#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Getting node %u from freelist[%u]\n", nodeIndex, m_freeOffset + 1);
#endif
	m_nodes[nodeIndex] = { dataOffset, size, topNodeIndex };
	if (topNodeIndex != Node::UNUSED)
		m_nodes[topNodeIndex].binListPrev = nodeIndex;
	m_binIndices[binIndex] = nodeIndex;

	m_freeStorage += size;
#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Free storage: %u (+%u) (insertNodeIntoBin)\n", m_freeStorage, size);
#endif

	return nodeIndex;
}

void OffsetAllocator::removeNodeFromBin(uint32 nodeIndex)
{
	const Node& node = m_nodes[nodeIndex];

	if (node.binListPrev != Node::UNUSED)
	{
		// Easy case: We have previous node. Just remove this node from the middle of the list.
		m_nodes[node.binListPrev].binListNext = node.binListNext;
		if (node.binListNext != Node::UNUSED)
			m_nodes[node.binListNext].binListPrev = node.binListPrev;
	}
	else
	{
		// Hard case: We are the first node in a bin. Find the bin.

		// Round down to bin index to ensure that bin >= alloc
		uint32 binIndex = uintToSmallFloatRoundDown(node.dataSize);

		uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
		uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

		m_binIndices[binIndex] = node.binListNext;
		if (node.binListNext != Node::UNUSED) m_nodes[node.binListNext].binListPrev = Node::UNUSED;

		// Bin empty?
		if (m_binIndices[binIndex] == Node::UNUSED)
		{
			// Remove a leaf bin mask bit
			m_usedBins[topBinIndex] &= ~(1 << leafBinIndex);

			// All leaf bins empty?
			if (m_usedBins[topBinIndex] == 0)
			{
				// Remove a top bin mask bit
				m_usedBinsTop &= ~(1 << topBinIndex);
			}
		}
	}

	// Insert the node to freelist
#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Putting node %u into freelist[%u] (removeNodeFromBin)\n", nodeIndex, m_freeOffset + 1);
#endif
	m_freeNodes[++m_freeOffset] = nodeIndex;

	m_freeStorage -= node.dataSize;
#ifdef DEBUG_VERBOSE
	printf("OffsetAllocator: Free storage: %u (-%u) (removeNodeFromBin)\n", m_freeStorage, node.dataSize);
#endif
}

uint32 OffsetAllocator::allocationSize(Alloc allocation) const
{
	if (allocation.metadata == Alloc::NO_SPACE)
		return 0;

	if (!m_nodes)
		return 0;

	return m_nodes[allocation.metadata].dataSize;
}

OffsetAllocator::StorageReport OffsetAllocator::storageReport() const
{
	// Out of allocations? -> Zero free space
	if (m_freeOffset == 0)
		return {};

	StorageReport report;
	report.totalFreeSpace = m_freeStorage;
	if (m_usedBinsTop)
	{
		uint32 topBinIndex = 31 - lzcnt_nonzero(m_usedBinsTop);
		uint32 leafBinIndex = 31 - lzcnt_nonzero(m_usedBins[topBinIndex]);
		report.largestFreeRegion = smallFloatToUint((topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex);
		ASSERT(report.totalFreeSpace >= report.largestFreeRegion);
	}

	return report;
}

OffsetAllocator::StorageReportFull OffsetAllocator::storageReportFull() const
{
	StorageReportFull report;
	for (uint32 i = 0; i < NUM_LEAF_BINS; i++)
	{
		uint32 count = 0;
		uint32 nodeIndex = m_binIndices[i];
		while (nodeIndex != Node::UNUSED)
		{
			nodeIndex = m_nodes[nodeIndex].binListNext;
			count++;
		}
		StorageReportFull::Region& region = report.freeRegions[i];
		region.size = smallFloatToUint(i);
		region.count = count;
	}
	return report;
}
}
#pragma once

namespace hptt {

   /**
    * \brief A ComputNode encodes a loop.
    */
class alignas(16) ComputeNode
{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), indexA(false), indexB(false), offDiffAB(std::numeric_limits<int>::min()), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   size_t start; //!< start index for at the current loop
   size_t end; //!< end index for at the current loop
   size_t inc; //!< increment for at the current loop
   size_t lda; //!< stride of A w.r.t. the loop index
   size_t ldb; //!< stride of B w.r.t. the loop index
   bool   indexA; //!< true if index of A is innermost (0)
   bool   indexB; //!< true if index of B is innermost (0)
   int    offDiffAB; //!< difference in offset A and B (i.e., A - B) at the current loop
   ComputeNode *next; //!< next ComputeNode, this might be another loop or 'nullptr' (i.e., indicating that the macro-kernel should be called)
};

}

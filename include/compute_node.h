#pragma once

namespace hptt {

   /**
    * \brief A ComputNode encodes a loop.
    */
class ComputeNode
{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   int start; //!< start index for at the current loop
   int end; //!< end index for at the current loop
   int inc; //!< increment for at the current loop
   int lda; //!< stride of A w.r.t. the loop index
   int ldb; //!< stride of B w.r.t. the loop index
   ComputeNode *next; //!< next ComputeNode, this might be another loop or 'nullptr' (i.e., indicating that the macro-kernel should be called)
};

}

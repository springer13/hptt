#pragma once

namespace hptt {

class ComputeNode
{
   public:
      ComputeNode() : start(-1), end(-1), inc(-1), lda(-1), ldb(-1), next(nullptr) {}

      ~ComputeNode() {
         if ( next != nullptr )
            delete next;
      }

   size_t start;
   size_t end;
   size_t inc;
   size_t lda;
   size_t ldb;
   ComputeNode *next;
};

}

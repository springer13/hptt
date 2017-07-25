#pragma once

#include <vector>

#include "plan.h"


namespace hptt {

class ComputeNode;

class Plan
{
   public:
      Plan() : rootNodes_(nullptr), numTasks_(0) { }

      Plan(std::vector<int>loopOrder, std::vector<int>numThreadsAtLoop);

      ~Plan();


      const ComputeNode* getRootNode_const(int threadId) const;
      ComputeNode* getRootNode(int threadId) const;
      int getNumTasks() const { return numTasks_; } 

      void print() const; 

   private:
      int numTasks_;
      std::vector<int> loopOrder_;
      std::vector<int> numThreadsAtLoop_;
      ComputeNode *rootNodes_;
};

}

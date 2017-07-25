#pragma once

#include <vector>

#include "plan.h"


namespace hptt {

class ComputeNode;

/**
 * \brief A plan encodes the execution of a tensor transposition.
 *
 * It stores the loop order and parallelizes each loop.
 */
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
      std::vector<int> loopOrder_; //!< loop order. For example, if \f$ B_{1,0,2} \gets A_{0,1,2}\f$. loopOrder_ = {1,0,2} denotes that B is travesed in a linear fashion.
      std::vector<int> numThreadsAtLoop_;
      ComputeNode *rootNodes_;
};

}

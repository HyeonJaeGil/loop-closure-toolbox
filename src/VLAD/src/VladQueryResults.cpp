#include "VladQueryResults.h"

namespace VLAD {

std::ostream &operator<<(std::ostream &os, const Result &ret) {
  os << "<EntryId: " << ret.id << ", Score: " << ret.score << ", Inliers: " << ret.inliers << ">";
  return os;
}

std::ostream &operator<<(std::ostream &os, const QueryResults &rets) {
  if (rets.size() == 1)
    os << "1 result:" << std::endl;
  else
    os << rets.size() << " results:" << std::endl;

  QueryResults::const_iterator rit;
  for (rit = rets.begin(); rit != rets.end(); ++rit) {
    os << *rit;
    if (rit + 1 != rets.end())
      os << std::endl;
  }
  return os;
}
} // namespace VLAD
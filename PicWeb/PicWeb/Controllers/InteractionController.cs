using PicWeb.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace PicWeb.Controllers
{
    public class InteractionController : BaseController
    {
        
        [HttpPost]
        public JsonResult ToggleLike(int postId)
        {
            if (Session["MemberId"] == null)
            {
                return Json(new { success = false, redirectUrl = Url.Action("Login", "Account") });
            }

            try
            {
                int memberId = (int)Session["MemberId"];
                var existingLike = db.LikeRecord.FirstOrDefault(l =>
                    l.PostID == postId && l.MemID == memberId);

                if (existingLike == null)
                {
                    // 新增讚
                    db.LikeRecord.Add(new LikeRecord
                    {
                        PostID = postId,
                        MemID = memberId,
                        LikeDate = DateTime.Now
                    });

                    // 更新貼文讚數
                    var post = db.Post.Find(postId);
                    post.LikeNum++;
                }
                else
                {
                    // 取消讚
                    db.LikeRecord.Remove(existingLike);
                    var post = db.Post.Find(postId);
                    post.LikeNum--;
                }

                db.SaveChanges();

                return Json(new
                {
                    success = true,
                    liked = (existingLike == null),
                    likeCount = db.Post.Find(postId).LikeNum
                });
            }
            catch
            {
                return Json(new { success = false });
            }
        }

        // POST: 新增留言
        [HttpPost]
        public JsonResult AddComment(int postId, string message)
        {
            if (Session["MemberId"] == null)
            {
                return Json(new { success = false, redirectUrl = Url.Action("Login", "Account") });
            }

            try
            {
                int memberId = (int)Session["MemberId"];
                var comment = new Comment
                {
                    PostID = postId,
                    MemID = memberId,
                    Message = message,
                    CommentDate = DateTime.Now
                };

                db.Comment.Add(comment);
                db.SaveChanges();

                return Json(new
                {
                    success = true,
                    commentId = comment.ID,
                    memberName = db.Member.Find(memberId).MemName,
                    date = comment.CommentDate.ToString("yyyy-MM-dd HH:mm")
                });
            }
            catch
            {
                return Json(new { success = false });
            }
        }

        // GET: 取得留言列表
        public ActionResult GetComments(int postId)
        {
            var comments = db.Comment
                .Where(c => c.PostID == postId)
                .OrderByDescending(c => c.CommentDate)
                .Select(c => new
                {
                    c.ID,
                    MemberName = c.Member.MemName,
                    c.Message,
                    c.CommentDate  // 先取得日期
                })
                .ToList()  // 先執行資料庫查詢
                .Select(c => new  // 再進行日期格式化
                {
                    c.ID,
                    c.MemberName,
                    c.Message,
                    Date = c.CommentDate.ToString("yyyy-MM-dd HH:mm")
                });

            return Json(comments, JsonRequestBehavior.AllowGet);
        }
    }
}
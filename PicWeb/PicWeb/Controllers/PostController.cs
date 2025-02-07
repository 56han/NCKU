using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using PicWeb.Models;

namespace PicWeb.Controllers
{
    public class PostController : BaseController
    {
        //PicWebDBEntities db = new PicWebDBEntities();

        public ActionResult Index()
        {
            // 只獲取圖片資訊
            var posts = db.Post
                .Select(p => new PostViewModel
                {
                    Id = p.ID,
                    Pic = p.Pic,
                    Note = p.Note
                })
                .ToList();

            ViewBag.PostCount = posts.Count;  // 總貼文數
            ViewBag.CameraCount = db.Camera.Count();  // 相機數量

            return View(posts);
        }

        public ActionResult AllPost()
        {
            ViewBag.CurrentPage = "AllPost";

            var dbPosts = db.Post
                .Include("Camera")
                .Include("Member")
                .OrderByDescending(p => p.PostDate)
                .ToList();

            var postViewModels = new List<PostViewModel>();

            int? currentMemberId = Session["MemberId"] as int?;

            foreach (var p in dbPosts)
            {
                postViewModels.Add(new PostViewModel
                {
                    Id = p.ID,
                    CamId = p.CamID,
                    MemId = p.MemID,
                    Note = p.Note,
                    Pic = p.Pic,
                    LikeNum = p.LikeNum,
                    PostDate = p.PostDate,
                    CamName = p.Camera.CamName,
                    MemName = p.Member.MemName,
                    IsLiked = currentMemberId.HasValue &&
                     db.LikeRecord.Any(l => l.PostID == p.ID &&
                                           l.MemID == currentMemberId.Value)
                });
            }

            return View(postViewModels);
        }

        public ActionResult Camera(string camName)
        {
            ViewBag.CurrentCamera = camName;

            var dbPosts = db.Post
                .Include("Camera")
                .Include("Member")
                .Where(p => p.Camera.CamName == camName)  // 篩選特定相機的貼文
                .OrderByDescending(p => p.PostDate)
                .ToList();

            var postViewModels = new List<PostViewModel>();

            int? currentMemberId = Session["MemberId"] as int?;

            foreach (var p in dbPosts)
            {
                postViewModels.Add(new PostViewModel
                {
                    Id = p.ID,
                    CamId = p.CamID,
                    MemId = p.MemID,
                    Note = p.Note,
                    Pic = p.Pic,
                    LikeNum = p.LikeNum,
                    PostDate = p.PostDate,
                    CamName = p.Camera.CamName,
                    MemName = p.Member.MemName,
                    IsLiked = currentMemberId.HasValue &&
                     db.LikeRecord.Any(l => l.PostID == p.ID &&
                                           l.MemID == currentMemberId.Value)
                });
            }

            ViewBag.CurrentCamera = camName;  // 將當前相機名稱傳給視圖
            return View(postViewModels);
        }
        public ActionResult Create()
        {
            // 檢查是否已登入
            if (Session["MemberId"] == null)
            {
                return RedirectToAction("Login", "Account");
            }

            // 取得相機清單供下拉選單使用
            ViewBag.CameraList = new SelectList(db.Camera, "ID", "CamName");
            return View();
        }

        [HttpPost]
        public ActionResult Create(Post post, HttpPostedFileBase imageFile, string NewCameraName)
        {
            if (Session["MemberId"] == null)
            {
                return RedirectToAction("Login", "Account");
            }

            if (post.CamID == 0 && !string.IsNullOrEmpty(NewCameraName))
            {
                // 檢查相機名稱是否已存在
                if (db.Camera.Any(c => c.CamName == NewCameraName))
                {
                    ModelState.AddModelError("", "此相機名稱已存在");
                    ViewBag.CameraList = new SelectList(db.Camera, "ID", "CamName");
                    return View(post);
                }

                // 新增相機
                var newCamera = new Camera { CamName = NewCameraName };
                db.Camera.Add(newCamera);
                db.SaveChanges();

                // 設定新相機ID
                post.CamID = newCamera.ID;
            }


            if (ModelState.IsValid && imageFile != null && imageFile.ContentLength > 0)
            {
                try
                {
                    int memberId = (int)Session["MemberId"];
                    string memberName = Session["MemberName"].ToString();
                    post.MemID = memberId;

                    int postCount = db.Post.Count(p => p.MemID == memberId) + 1;
                    string newFileName = $"{memberName}_{postCount}.jpeg";
                    string savePath = Path.Combine(Server.MapPath("~/Images"), newFileName);

                    // 使用 System.Drawing 處理圖片
                    using (var image = System.Drawing.Image.FromStream(imageFile.InputStream))
                    {
                        image.Save(savePath, System.Drawing.Imaging.ImageFormat.Jpeg);
                    }

                    post.Pic = newFileName;
                    post.PostDate = DateTime.Now;
                    post.LikeNum = 0;

                    db.Post.Add(post);
                    db.SaveChanges();

                    return RedirectToAction("AllPost");
                }
                catch (Exception ex)
                {
                    ModelState.AddModelError("", "上傳圖片時發生錯誤: " + ex.Message);
                    ViewBag.CameraList = new SelectList(db.Camera, "ID", "CamName", post.CamID);
                    return View(post);
                }
            }

            if (imageFile == null || imageFile.ContentLength == 0)
            {
                ModelState.AddModelError("", "請選擇要上傳的圖片");
            }

            ViewBag.CameraList = new SelectList(db.Camera, "ID", "CamName", post.CamID);
            return View(post);
        }

        public ActionResult UserPosts(int? id, int? scrollTo = null)
        {
            if (!id.HasValue)  // 加入檢查
            {
                return RedirectToAction("AllPost");
            }

            var dbPosts = db.Post
                .Include("Camera")
                .Include("Member")
                .Where(p => p.MemID == id.Value)  
                .OrderByDescending(p => p.PostDate)  
                .ToList();

            var postViewModels = new List<PostViewModel>();
            int? currentMemberId = Session["MemberId"] as int?;

            foreach (var p in dbPosts)
            {
                postViewModels.Add(new PostViewModel
                {
                    Id = p.ID,
                    CamId = p.CamID,
                    MemId = p.MemID,
                    Note = p.Note,
                    Pic = p.Pic,
                    LikeNum = p.LikeNum,
                    PostDate = p.PostDate,
                    CamName = p.Camera.CamName,
                    MemName = p.Member.MemName,
                    IsLiked = currentMemberId.HasValue &&
                             db.LikeRecord.Any(l => l.PostID == p.ID &&
                                                  l.MemID == currentMemberId.Value)
                });
            }

            ViewBag.ScrollTo = scrollTo;

            // 傳遞用戶名稱到 ViewBag
            if (dbPosts.Any())
            {
                ViewBag.UserName = dbPosts.First().Member.MemName;
            }

            
            return View(postViewModels);
        }
    }
}
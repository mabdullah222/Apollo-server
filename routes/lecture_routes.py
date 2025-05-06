from flask import Blueprint, request, jsonify, send_file
import asyncio
import os
import uuid
import logging
from datetime import datetime
from prisma import Prisma


from workflows.PresentationWorkflow import PresentationFlow

logger = logging.getLogger(__name__)
router = Blueprint("lecture", __name__)

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "lecVids")
os.makedirs(VIDEOS_DIR, exist_ok=True)

db = Prisma()

def create_workflow():
    return PresentationFlow().app

@router.route("/generate-lecture", methods=["POST"])
def generate_lecture():
    data = request.get_json()

    async def process():
        try:
            await db.connect()

            workflow = create_workflow()

            initial_state = {
                "topic": data["topic"],
                "toc": [],
                "resources": [],
                "documents": [],
                "vector_db": "",
                "content": {},
                "slides": [],
                "lecture": [],
                "video_paths": []
            }

            final_state = await asyncio.to_thread(workflow.invoke, initial_state)

            lecture_in_db=await db.lecture.create(data={
                "topic": data["topic"],
                "toc": final_state.get("toc", []),
                "lecture": final_state["lecture"],
                "vector_db": final_state["vector_db"],
                "video_paths": final_state.get("video_paths"),
                "completed": False,
                "resources": final_state.get("resources"),
                "created_at": datetime.now(),
                "userId": data["clerkUserId"],
                "progress": 0,
            })

            for slide in final_state['slides']:
                await db.slide.create(data={'title':slide['title'],"lectureId":lecture_in_db.id,"content":slide['content'],"code":slide['code']})

            return {
                "lecture_id": lecture_in_db.id,
                "status": "completed",
                "progress": 0,
                "message": "Lecture generation completed successfully"
            }

        finally:
            await db.disconnect()

    return jsonify(asyncio.run(process()))

@router.route("/lecture-status/<lecture_id>", methods=["GET"])
def lecture_status(lecture_id):
    async def process():
        await db.connect()
        lecture = await db.lecture.find_unique(where={"id": lecture_id},include={"slide":True})
        await db.disconnect()

        if not lecture:
            return jsonify({"error": "Lecture not found"}), 404

        return {
            "lecture_id": lecture.id,
            "completed": lecture.completed,
            "video_paths": lecture.video_paths,
            "slides": [{"title":slide.title,"content":slide.content,"code":slide.code} for slide in lecture.slide],
            "lecture": lecture.lecture,
            "progress": lecture.progress,
            "topic": lecture.topic,
            "resources": lecture.resources,
            "vector_db": lecture.vector_db,
        }

    return jsonify(asyncio.run(process()))

@router.route("/lectures", methods=["GET"])
def get_all_lectures():
    async def process():
        await db.connect()
        lectures = await db.lecture.find_many()
        await db.disconnect()

        return [
            {
                "lecture_id": lec.id,
                "completed": lec.completed,
                "slides": lec.slides,
                "lecture": lec.lecture,
                "progress": lec.progress,
            }
            for lec in lectures
        ]

    return jsonify(asyncio.run(process()))

@router.route("/lecture/<lecture_id>", methods=["DELETE"])
def delete_lecture(lecture_id):
    async def process():
        await db.connect()
        lecture = await db.lecture.find_unique(where={"id": lecture_id})

        if not lecture:
            await db.disconnect()
            return jsonify({"error": "Lecture not found"}), 404

        for video_path in lecture.video_paths:
            if os.path.exists(video_path):
                os.remove(video_path)

        await db.lecture.delete(where={"id": lecture_id})
        await db.disconnect()

        return {"status": "success", "message": "Lecture deleted"}

    return jsonify(asyncio.run(process()))

@router.route("/register",methods=["POST"])
def register_user():
    data = request.get_json()

    async def process():
        await db.connect()
        user = await db.user.create(data={
            "clerkuserId": data["clerkUserId"],
            "name": data["name"],
            "email": data["email"]
        })
        await db.disconnect()

        return {"user_id": user.id, "status": "registered"}

    return jsonify(asyncio.run(process()))



@router.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "version": "1.0.0"})

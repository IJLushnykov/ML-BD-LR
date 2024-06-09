db.Tech.find().pretty()

db.Tech.countDocuments({ category: "TV" })

db.Tech.distinct("category").length

db.Tech.distinct("producer")

db.Tech.find({
    $and: [
        { category: "TV" },
        { price: { $gte: 600, $lte: 975} }
    ]
}).pretty()

db.Tech.find({
    $or: [
        { model: "Samsung Tv" },
        { model: "Samsung Smartest TV" }
    ]
}).pretty()

db.Tech.find({
    producer: { $in: ["Apple", "Sony", "LG"] }
}).pretty()

db.Tech.updateMany(
    { category: "TV" },
    {
        $set: { "Size": "34''" },
        $currentDate: { "Updated at": true }
    }
)

db.Tech.find({ "Size": { $exists: true } }).pretty()

db.Tech.updateMany(
    { "Size": { $exists: true } },
    { $inc: { "price": 20 } }
)
db.Tech.find({ "Size": { $exists: true } }).pretty()
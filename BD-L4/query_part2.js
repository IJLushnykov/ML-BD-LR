db.orders.insertMany([
    {
        "order_number": 201513,
        "date": ISODate("2024-04-14"),
        "total_sum": 1720.0,
        "customer": {
            "name": "Andrii",
            "surname": "Rodinov",
            "phones": [1234567],
            "address": "Peremohy 37, Kyiv, UA"
        },
        "payment": {
            "card_owner": "Andrii Rodionov",
            "cardId": 12345678
        },
        "items_id": ["66526f3d6b3cf577f8c748c1", "66526f3d6b3cf577f8c748c5"]
    },
    {
        "order_number": 201514,
        "date": ISODate("2024-05-20"),
        "total_sum": 2500.0,
        "customer": {
            "name": "Ivan",
            "surname": "Lushnykov",
            "phones": [+313456789, +3809876543],
            "address": "Shevchenko 10, Kyiv, UA"
        },
        "payment": {
            "card_owner": "Ivan Lushnykov",
            "cardId": 87654321
        },
        "items_id": ["66526f3d6b3cf577f8c748c3", "66526f3d6b3cf577f8c748c4", "66526f3d6b3cf577f8c748c5"]
    },
    {
        "order_number": 201515,
        "date": ISODate("2024-05-15"),
        "total_sum": 1150.0,
        "customer": {
            "name": "Andrii",
            "surname": "Rodinov",
            "phones": [06234567],
            "address": "Peremohy 37, Kyiv, UA"
        },
        "payment": {
            "card_owner": "Andrii Rodinov",
            "cardId": 12345678
        },
        "items_id": ["66526f3d6b3cf577f8c748c0", "66526f3d6b3cf577f8c748c6"]
    }
]);

db.Tech.find().pretty()
db.orders.find().pretty()

db.orders.find({ total_sum: { $gt: 1500 } }).pretty()

db.orders.find({ "customer.name": "Andrii", "customer.surname": "Rodinov" }).pretty()

db.orders.find({ items_id: "66526f3d6b3cf577f8c748c6" }).pretty()

db.orders.updateMany(
    { items_id:"66526f3d6b3cf577f8c748c5" },
    {
        $push: { items_id: "66526f3d6b3cf577f8c748c2" },
        $inc: { total_sum: 800 }
    }
)

db.orders.findOne(
    { order_number: 201514 },
    { _id: 0, items_count: { $size: "$items_id" } }
)

db.orders.find(
    { total_sum: { $gt: 1900 } },
    { _id: 0, customer: 1, "payment.cardId": 1 }
).pretty()

db.orders.updateMany(
    { date: { $gte: ISODate("2024-04-01"), $lte: ISODate("2024-05-26") } },
    { $pull: { items_id: ObjectId("66526f3d6b3cf577f8c748c3") } }
)

db.orders.updateMany(
    { "customer.name": "Andrii", "customer.surname": "Rodinov" },
    { $set: { "customer.surname": "Rodinin" } }
)


db.orders.aggregate([
    { $match: { "customer.name": "Andrii", "customer.surname": "Rodinin" } },
    { $lookup: {
        from: "Tech",
        localField: "items_id",
        foreignField: "_id",
        as: "items_info"
    }},
    { $project: {
        _id: "66526f3d6b3cf577f8c748c6" ,
        customer: 1,
        items_info: { producer: "Apple"}
    }}
]).pretty()